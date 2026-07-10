//! Policy network: tch port of `rl/policy.py`.
//!
//! v1 deviation (documented, temporary): there is no exported AE checkpoint
//! in this repo yet (`ae/model_v3.py` is trained separately and frozen for
//! PPO), so the grid the policy sees here is the *raw* pooled ego-class
//! occupancy + static-structure + transient planes instead of the AE's
//! 32ch latent - i.e. `ofcore::feat`'s `stat`/`transient` planes plus
//! `pool_ego_db`'s ego/defense-bonus fractions, all already at /REGION
//! resolution. `C_GRID` below reflects that (6 + 3 + 1 + 53 = 63 instead
//! of the real arch's 32 + 3 + 1 + 53 = 89). Swap in a real
//! `ofcore`-side AE encode() port once `scripts/export_safetensors.py`
//! has produced a checkpoint to validate against - the head/trunk
//! topology and factorized-action-head logic below already match
//! `Policy.forward`/`act`/`evaluate` field for field.
//!
//! v1 simplification: no fine/coarse foveation crop (`_fine_coverage` in
//! `rl/obs.py`) - every obs uses the *whole* /REGION grid as "fine" with
//! an all-ones coverage channel and a 2x avg-pooled "coarse" derived from
//! it, exactly matching `Policy._ensure_foveated`'s legacy fallback path
//! (a real, already-existing code path in the Python model, not an
//! invented shortcut). `fine_origin` is therefore always (0, 0).

use ofcore::feat::{ACTIONS, GW_MAX};
use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

pub const N_ACTIONS: i64 = ofcore::feat::N_ACTIONS as i64;
pub const MAX_SLOTS: i64 = ofcore::feat::MAX_SLOTS as i64;

pub const N_STATIC: i64 = 6;
pub const N_TRANSIENT: i64 = 53;
pub const C_GRID: i64 = N_STATIC + 3 + 1 + N_TRANSIENT; // 63
pub const C_GRID_FINE: i64 = C_GRID + 1; // 64
pub const N_LOCAL: i64 = 5;
pub const LOCAL: i64 = 64;
pub const P_FEAT: i64 = 21;
pub const N_SCALARS: i64 = 11;
pub const N_BUILD: i64 = 7;
pub const N_NUKE: i64 = 5;

/// `--foveate`: fixed output size (both dims) of the real fine-grid crop
/// window (see `PolicyNet::foveate`'s doc). Chosen well below typical
/// map sizes (`GW_MAX`=250, `GH_MAX`=150 in /REGION units) so the fine
/// tower's per-forward-pass cost stops scaling with map size; small
/// enough to keep the resolution genuinely "foveated" (a local, high-
/// detail view) rather than most of the map.
pub const FOVEATE_SIZE: i64 = 48;

pub const HIDDEN: i64 = 512;
pub const GC: i64 = 256;
pub const PC: i64 = 128;
pub const BLOCKS: i64 = 4;
pub const LC: i64 = 64;
pub const N_HEAD: i64 = 4;
pub const TF_FF: i64 = 2 * PC;
pub const TF_LAYERS: i64 = 2;

const MASKED_NEG: f64 = -1e9;

const NEEDS_PLAYER: &[&str] = &[
    "attack", "alliance_request", "alliance_reject", "break_alliance", "donate_gold",
    "donate_troops", "embargo", "retreat", "embargo_stop", "target_player", "alliance_extension",
];
const NEEDS_TILE: &[&str] = &[
    "boat", "build", "launch_nuke", "spawn", "upgrade_structure", "move_warship", "cancel_boat",
    "delete_unit",
];
const REFINE_TILE: &[&str] =
    &["spawn", "build", "upgrade_structure", "cancel_boat", "delete_unit"];
const NEEDS_QUANTITY: &[&str] = &["attack", "expand", "boat", "donate_gold", "donate_troops"];

pub fn needs_player(action_name: &str) -> bool {
    NEEDS_PLAYER.contains(&action_name)
}
pub fn needs_tile(action_name: &str) -> bool {
    NEEDS_TILE.contains(&action_name)
}
pub fn needs_quantity(action_name: &str) -> bool {
    NEEDS_QUANTITY.contains(&action_name)
}

fn action_table(names: &[&str], device: Device) -> Tensor {
    let v: Vec<f32> = ACTIONS.iter().map(|a| if names.contains(a) { 1.0 } else { 0.0 }).collect();
    Tensor::from_slice(&v).to_device(device)
}

/// Batched observation tensors. `grid`/`grid_valid`/`legal_tile` are the
/// full /REGION-resolution "fine" inputs (see module doc); coarse tensors
/// are derived from them inside `trunk_forward`.
pub struct Obs {
    pub grid: Tensor,          // (B, C_GRID, gh, gw) f32
    pub grid_valid: Tensor,    // (B, gh, gw) f32, all-ones in v1
    pub legal_tile: Tensor,    // (B, gh, gw) f32
    pub players: Tensor,       // (B, MAX_SLOTS, P_FEAT) f32
    pub pmask: Tensor,         // (B, MAX_SLOTS) f32
    pub local: Tensor,         // (B, N_LOCAL, LOCAL, LOCAL) f32
    pub scalars: Tensor,       // (B, N_SCALARS) f32
    pub legal_actions: Tensor, // (B, N_ACTIONS) f32
    pub legal_ptarget: Tensor, // (B, N_ACTIONS, MAX_SLOTS) f32
    pub legal_build: Tensor,   // (B, N_BUILD) f32
    pub legal_nuke: Tensor,    // (B, N_NUKE) f32
}

impl Obs {
    /// GPU-side row gather (no host round trip) - lets the training loop
    /// build the full rollout's `Obs` once per update and slice out each
    /// minibatch from tensors already resident on the shard's device,
    /// instead of re-running `batch::build_obs`'s CPU repack + host->device
    /// upload for every (epoch, minibatch) pair (see DEVLOG: that
    /// redundant re-upload of the (B, C_GRID, gh, gw) grid, tens of MB per
    /// sample, was the actual bottleneck behind "training phase" wall-time
    /// not translating into real `nvidia-smi` GPU utilization).
    pub fn index_select(&self, idx: &Tensor) -> Obs {
        Obs {
            grid: self.grid.index_select(0, idx),
            grid_valid: self.grid_valid.index_select(0, idx),
            legal_tile: self.legal_tile.index_select(0, idx),
            players: self.players.index_select(0, idx),
            pmask: self.pmask.index_select(0, idx),
            local: self.local.index_select(0, idx),
            scalars: self.scalars.index_select(0, idx),
            legal_actions: self.legal_actions.index_select(0, idx),
            legal_ptarget: self.legal_ptarget.index_select(0, idx),
            legal_build: self.legal_build.index_select(0, idx),
            legal_nuke: self.legal_nuke.index_select(0, idx),
        }
    }
}

/// Choice tensors for `evaluate()`: long fields use -1 where unused,
/// `quantity_frac` uses -1.0. Mirrors the dict `Policy.evaluate` expects.
pub struct ChoiceBatch {
    pub action: Tensor,        // (B,) i64
    pub player_slot: Tensor,   // (B,) i64, -1 unused
    pub tile_region: Tensor,   // (B,) i64, -1 unused
    pub build_type: Tensor,    // (B,) i64, -1 unused
    pub nuke_type: Tensor,     // (B,) i64, -1 unused
    pub quantity_frac: Tensor, // (B,) f32, -1.0 unused
}

impl ChoiceBatch {
    pub fn index_select(&self, idx: &Tensor) -> ChoiceBatch {
        ChoiceBatch {
            action: self.action.index_select(0, idx),
            player_slot: self.player_slot.index_select(0, idx),
            tile_region: self.tile_region.index_select(0, idx),
            build_type: self.build_type.index_select(0, idx),
            nuke_type: self.nuke_type.index_select(0, idx),
            quantity_frac: self.quantity_frac.index_select(0, idx),
        }
    }
}

fn categorical_sample(logits: &Tensor, greedy: bool) -> (Tensor, Tensor) {
    let logp_all = logits.log_softmax(-1, Kind::Float);
    let idx = if greedy {
        logits.argmax(-1, false)
    } else {
        let probs = logits.softmax(-1, Kind::Float);
        probs.multinomial(1, true).squeeze_dim(-1)
    };
    let logp = logp_all.gather(-1, &idx.unsqueeze(-1), false).squeeze_dim(-1);
    (idx, logp)
}

fn categorical_logp(logits: &Tensor, idx_clamped: &Tensor) -> Tensor {
    logits.log_softmax(-1, Kind::Float).gather(-1, &idx_clamped.unsqueeze(-1), false).squeeze_dim(-1)
}

fn categorical_entropy(logits: &Tensor) -> Tensor {
    let logp = logits.log_softmax(-1, Kind::Float);
    let p = logp.exp();
    -(p * logp).sum_dim_intlist(-1, false, Kind::Float)
}

/// alpha/beta >= 1 (unimodal, bounded Beta): 1 + softplus(raw).
fn quantity_ab(params: &Tensor) -> (Tensor, Tensor) {
    let ab = params.softplus() + 1.0;
    (ab.select(-1, 0), ab.select(-1, 1))
}

fn beta_log_prob(x: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    let lbeta: Tensor = a.lgamma() + b.lgamma() - (a + b).lgamma();
    let one_minus_x: Tensor = x.neg() + 1.0;
    (a - 1.0) * x.log() + (b - 1.0) * one_minus_x.log() - lbeta
}

fn beta_entropy(a: &Tensor, b: &Tensor) -> Tensor {
    let ab: Tensor = a + b;
    let lbeta: Tensor = a.lgamma() + b.lgamma() - ab.lgamma();
    let da = a.digamma();
    let db = b.digamma();
    let dab = ab.digamma();
    lbeta - (a - 1.0) * da - (b - 1.0) * db + (ab - 2.0) * dab
}

/// Manual mixed-precision conv2d: casts the layer's (f32, VarStore-owned)
/// weight/bias to bf16 on the fly and runs the conv in bf16, without ever
/// mutating the stored f32 parameters (those stay f32 for the optimizer -
/// see module doc / `--amp` in `main.rs`). `x` is expected to already be
/// bf16 (callers cast once at the tower's input boundary and keep the
/// whole chain in bf16, matching how real autocast keeps a run of
/// non-numerically-sensitive ops - conv/silu/add here - all in the reduced
/// dtype instead of round-tripping through f32 after every op).
fn conv2d_bf16(conv: &nn::Conv2D, x: &Tensor, stride: [i64; 2], padding: [i64; 2]) -> Tensor {
    let ws = conv.ws.to_kind(Kind::BFloat16);
    let bs = conv.bs.as_ref().map(|b| b.to_kind(Kind::BFloat16));
    x.conv2d(&ws, bs.as_ref(), stride, padding, [1i64, 1], 1)
}

struct ResBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
}

impl ResBlock {
    fn new(p: &nn::Path, c: i64) -> Self {
        let cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        ResBlock { conv1: nn::conv2d(p / "conv1", c, c, 3, cfg), conv2: nn::conv2d(p / "conv2", c, c, 3, cfg) }
    }
    fn forward(&self, x: &Tensor, amp: bool) -> Tensor {
        if amp {
            let h = conv2d_bf16(&self.conv1, x, [1, 1], [1, 1]).silu();
            (x + conv2d_bf16(&self.conv2, &h, [1, 1], [1, 1])).silu()
        } else {
            let h = self.conv1.forward(x).silu();
            (x + self.conv2.forward(&h)).silu()
        }
    }
}

struct GridTower {
    stem: nn::Conv2D,
    blocks: Vec<ResBlock>,
}

impl GridTower {
    fn new(p: &nn::Path, c_in: i64, gc: i64, blocks: i64) -> Self {
        let cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        let stem = nn::conv2d(p / "stem", c_in, gc, 3, cfg);
        let blocks = (0..blocks).map(|i| ResBlock::new(&(p / "block" / i), gc)).collect();
        GridTower { stem, blocks }
    }
    /// `amp=true` runs the whole tower (stem + every residual block) in
    /// bf16, casting in once at the input and back to f32 once at the
    /// output (see `conv2d_bf16`); `amp=false` is the byte-for-byte
    /// original f32 path.
    fn forward(&self, x: &Tensor, amp: bool) -> Tensor {
        if amp {
            let xb = x.to_kind(Kind::BFloat16);
            let mut h = conv2d_bf16(&self.stem, &xb, [1, 1], [1, 1]).silu();
            for b in &self.blocks {
                h = b.forward(&h, true);
            }
            h.to_kind(Kind::Float)
        } else {
            let mut h = self.stem.forward(x).silu();
            for b in &self.blocks {
                h = b.forward(&h, false);
            }
            h
        }
    }
}

struct LocalNet {
    c1: nn::Conv2D,
    c2: nn::Conv2D,
    c3: nn::Conv2D,
}

impl LocalNet {
    fn new(p: &nn::Path) -> Self {
        let cfg = |pad| nn::ConvConfig { padding: pad, stride: 2, ..Default::default() };
        LocalNet {
            c1: nn::conv2d(p / "c1", N_LOCAL, 32, 3, cfg(1)),
            c2: nn::conv2d(p / "c2", 32, 64, 3, cfg(1)),
            c3: nn::conv2d(p / "c3", 64, LC, 3, cfg(1)),
        }
    }
    fn forward(&self, x: &Tensor, amp: bool) -> Tensor {
        if amp {
            let xb = x.to_kind(Kind::BFloat16);
            let h = conv2d_bf16(&self.c1, &xb, [2, 2], [1, 1]).silu();
            let h = conv2d_bf16(&self.c2, &h, [2, 2], [1, 1]).silu();
            let h = conv2d_bf16(&self.c3, &h, [2, 2], [1, 1]).silu();
            h.adaptive_avg_pool2d([1, 1]).flatten(1, -1).to_kind(Kind::Float)
        } else {
            let h = self.c1.forward(x).silu();
            let h = self.c2.forward(&h).silu();
            let h = self.c3.forward(&h).silu();
            h.adaptive_avg_pool2d([1, 1]).flatten(1, -1)
        }
    }
}

/// Hand-rolled `nn.TransformerEncoderLayer` (batch_first, post-norm,
/// dropout=0) since tch has no built-in transformer module.
struct EncoderLayer {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    out: nn::Linear,
    ln1: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    ln2: nn::LayerNorm,
}

impl EncoderLayer {
    fn new(p: &nn::Path, d: i64, ff: i64) -> Self {
        EncoderLayer {
            q: nn::linear(p / "q", d, d, Default::default()),
            k: nn::linear(p / "k", d, d, Default::default()),
            v: nn::linear(p / "v", d, d, Default::default()),
            out: nn::linear(p / "out", d, d, Default::default()),
            ln1: nn::layer_norm(p / "ln1", vec![d], Default::default()),
            ff1: nn::linear(p / "ff1", d, ff, Default::default()),
            ff2: nn::linear(p / "ff2", ff, d, Default::default()),
            ln2: nn::layer_norm(p / "ln2", vec![d], Default::default()),
        }
    }

    /// x: (B, S, D); key_pad_bias: (B, 1, 1, S) additive bias, -1e9 at
    /// padded keys.
    fn forward(&self, x: &Tensor, key_pad_bias: &Tensor) -> Tensor {
        let (b, s, d) = x.size3().unwrap();
        let hd = d / N_HEAD;
        let split = |t: &Tensor| t.view([b, s, N_HEAD, hd]).permute([0, 2, 1, 3]); // (B, H, S, hd)
        let q = split(&self.q.forward(x));
        let k = split(&self.k.forward(x));
        let v = split(&self.v.forward(x));
        let scores = q.matmul(&k.transpose(-2, -1)) / (hd as f64).sqrt();
        let scores = scores + key_pad_bias;
        let attn = scores.softmax(-1, Kind::Float).matmul(&v); // (B, H, S, hd)
        let attn = attn.permute([0, 2, 1, 3]).contiguous().view([b, s, d]);
        let x = self.ln1.forward(&(x + self.out.forward(&attn)));
        let ff = self.ff2.forward(&self.ff1.forward(&x).relu());
        self.ln2.forward(&(&x + ff))
    }
}

/// Everything derived from the crop decision: the (possibly cropped)
/// fine-grid inputs plus the always-full-map coarse-grid inputs, and
/// the per-sample crop bounds needed to translate fine-local tile picks
/// back to absolute map coordinates (see `PolicyNet::foveate`'s doc).
struct Foveation {
    grid_fine: Tensor,       // (B, C_GRID_FINE, fine_h, fine_w)
    legal_tile_fine: Tensor, // (B, fine_h, fine_w)
    grid_valid_fine: Tensor, // (B, fine_h, fine_w) - 0 in the crop's padded region, else 1
    grid_coarse: Tensor,     // (B, C_GRID, cgh, cgw) - always the whole map, unaffected by the crop
    gc_valid: Tensor,        // (B, cgh, cgw)
    legal_tile_coarse: Tensor, // (B, cgh, cgw)
    /// Legal+valid fine cells, projected into full-map coarse space: the
    /// whole map when `!foveate` (matches the legacy fallback exactly -
    /// see `PolicyNet::fine_to_coarse_mask`), or just the crop window's
    /// footprint (zero elsewhere) when `foveate` is on.
    fine_coarse: Tensor, // (B, cgh, cgw)
    origin_y: Tensor,    // (B,) i64 - crop top-left row, absolute grid coords (0 when `!foveate`)
    origin_x: Tensor,    // (B,) i64
    fine_h: i64,
    fine_w: i64,
}

pub struct PolicyNet {
    grid_coarse_net: GridTower,
    grid_fine_net: GridTower,
    local_net: LocalNet,
    player_in: nn::Linear,
    tf_layers: Vec<EncoderLayer>,
    trunk1: nn::Linear,
    trunk2: nn::Linear,
    head_action: nn::Linear,
    head_player_q: nn::Linear,
    head_tile_coarse: (nn::Conv2D, nn::Conv2D),
    head_tile_fine: (nn::Conv2D, nn::Conv2D),
    head_build: nn::Linear,
    head_nuke: nn::Linear,
    head_quantity: nn::Linear,
    head_value: nn::Linear,
    device: Device,
    /// `--amp`: run the conv-heavy submodules (grid towers, local net,
    /// tile heads) in manually-managed bf16 instead of f32 (see
    /// `conv2d_bf16`'s doc comment - tch-rs 0.24 has no dtype-selectable
    /// autocast context, see DEVLOG). Weights/optimizer state/final
    /// logits stay f32 regardless.
    amp: bool,
    /// `--foveate`: crop the fine grid branch to a small
    /// `FOVEATE_SIZE`x`FOVEATE_SIZE` window centered on the agent's own
    /// tile centroid instead of using the whole map (see
    /// `PolicyNet::foveate`'s doc). Off by default (matches the existing
    /// legacy fallback - see module doc).
    foveate: bool,
}

impl PolicyNet {
    pub fn new(vs: &nn::Path, amp: bool, foveate: bool) -> Self {
        let conv1 = |p: &nn::Path, ci, co| nn::conv2d(p, ci, co, 1, Default::default());
        PolicyNet {
            grid_coarse_net: GridTower::new(&(vs / "grid_coarse"), C_GRID, GC, BLOCKS),
            grid_fine_net: GridTower::new(&(vs / "grid_fine"), C_GRID_FINE, GC, BLOCKS),
            local_net: LocalNet::new(&(vs / "local")),
            player_in: nn::linear(vs / "player_in", P_FEAT, PC, Default::default()),
            tf_layers: (0..TF_LAYERS).map(|i| EncoderLayer::new(&(vs / "tf" / i), PC, TF_FF)).collect(),
            trunk1: nn::linear(vs / "trunk1", 2 * GC + PC + LC + N_SCALARS, HIDDEN, Default::default()),
            trunk2: nn::linear(vs / "trunk2", HIDDEN, HIDDEN, Default::default()),
            head_action: nn::linear(vs / "head_action", HIDDEN, N_ACTIONS, Default::default()),
            head_player_q: nn::linear(vs / "head_player_q", HIDDEN, PC, Default::default()),
            head_tile_coarse: (
                conv1(&(vs / "htc1"), GC + HIDDEN, 256),
                conv1(&(vs / "htc2"), 256, 1),
            ),
            head_tile_fine: (
                conv1(&(vs / "htf1"), GC + HIDDEN, 256),
                conv1(&(vs / "htf2"), 256, 1),
            ),
            head_build: nn::linear(vs / "head_build", HIDDEN, N_BUILD, Default::default()),
            head_nuke: nn::linear(vs / "head_nuke", HIDDEN, N_NUKE, Default::default()),
            head_quantity: nn::linear(vs / "head_quantity", HIDDEN, 2, Default::default()),
            head_value: nn::linear(vs / "head_value", HIDDEN, 1, Default::default()),
            device: vs.device(),
            amp,
            foveate,
        }
    }

    /// Derives fine/coarse grid + validity + legal-tile tensors from the
    /// single full-res `grid`/`grid_valid`/`legal_tile` inputs.
    ///
    /// `use_crop=false` matches `Policy._ensure_foveated`'s legacy
    /// fallback exactly (see module doc): fine=full map, coarse=2x
    /// avg-pooled full map, all-ones coverage, `fine_origin`=(0,0).
    ///
    /// `use_crop=true` is the real foveated crop: the fine branch is a
    /// fixed `FOVEATE_SIZE`x`FOVEATE_SIZE` window gathered (not resized -
    /// a hard crop) from the full-res grid, with a per-sample origin
    /// centered on the agent's own tile centroid (computed from the
    /// "mine" ego-occupancy channel already in `o.grid`, channel index
    /// `N_STATIC` - see `vecenv.rs::prepare`'s channel layout and
    /// `ofcore::feat::pool_ego_db`; falls back to the map center before
    /// the agent owns anything, e.g. spawn phase, mirroring
    /// `ofcore::feat::local_crop`'s identical fallback for the unrelated
    /// LocalNet branch). The origin is snapped to an even coordinate so
    /// the crop's fine cells stay aligned to the coarse grid's 2x2
    /// blocks (every downstream coarse<->fine mapping assumes that). If
    /// the map is smaller than `FOVEATE_SIZE` in either dimension, the
    /// crop covers the whole map in that dimension and the remainder is
    /// zero-padded (`grid_valid_fine`/`legal_tile_fine` are 0 there, so
    /// the padded cells are never legal to pick).
    fn foveate(o: &Obs, use_crop: bool) -> Foveation {
        let (b, gh, gw) = o.legal_tile.size3().unwrap();
        let grid_coarse = o.grid.avg_pool2d([2, 2], [2, 2], [0, 0], true, false, None::<i64>);
        let gc_valid =
            o.grid_valid.unsqueeze(1).max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true).squeeze_dim(1);
        let legal_tile_coarse =
            o.legal_tile.unsqueeze(1).max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true).squeeze_dim(1);

        if !use_crop {
            let grid_fine = Tensor::cat(&[&o.grid, &o.grid_valid.unsqueeze(1)], 1);
            let fine_coarse = Self::fine_to_coarse_mask(o);
            let device = o.grid.device();
            let origin_y = Tensor::zeros([b], (Kind::Int64, device));
            let origin_x = origin_y.zeros_like();
            return Foveation {
                grid_fine,
                legal_tile_fine: o.legal_tile.shallow_clone(),
                grid_valid_fine: o.grid_valid.shallow_clone(),
                grid_coarse,
                gc_valid,
                legal_tile_coarse,
                fine_coarse,
                origin_y,
                origin_x,
                fine_h: gh,
                fine_w: gw,
            };
        }

        debug_assert!(gh >= 2 && gw >= 2, "foveate crop needs a grid at least 2x2, got ({gh}, {gw})");
        let fine_h = (FOVEATE_SIZE.min(gh)).max(2);
        let fine_w = (FOVEATE_SIZE.min(gw)).max(2);
        let fine_h = fine_h - fine_h % 2;
        let fine_w = fine_w - fine_w % 2;
        let mine = o.grid.select(1, N_STATIC); // (B, gh, gw): own-tile occupancy fraction
        let (origin_y, origin_x) = crop_origin(&mine, gh, gw, fine_h, fine_w);

        let grid_cropped =
            crop_and_pad(&o.grid, &origin_y, &origin_x, fine_h, fine_w, FOVEATE_SIZE, FOVEATE_SIZE);
        let grid_valid_fine = crop_and_pad(
            &o.grid_valid.unsqueeze(1),
            &origin_y,
            &origin_x,
            fine_h,
            fine_w,
            FOVEATE_SIZE,
            FOVEATE_SIZE,
        )
        .squeeze_dim(1);
        let legal_tile_fine = crop_and_pad(
            &o.legal_tile.unsqueeze(1),
            &origin_y,
            &origin_x,
            fine_h,
            fine_w,
            FOVEATE_SIZE,
            FOVEATE_SIZE,
        )
        .squeeze_dim(1);
        let grid_fine = Tensor::cat(&[&grid_cropped, &grid_valid_fine.unsqueeze(1)], 1);
        let (cgh, cgw) = (gc_valid.size()[1], gc_valid.size()[2]);
        // `legal_tile_fine`/`grid_valid_fine` are already crop+pad'd to
        // FOVEATE_SIZE with zeros outside the true (fine_h, fine_w) crop
        // (see `crop_and_pad`'s doc), so the mask this builds is correct
        // over the full FOVEATE_SIZE window, not just the pre-pad crop.
        let fine_coarse = fine_to_coarse_mask_cropped(
            &legal_tile_fine,
            &grid_valid_fine,
            &origin_y,
            &origin_x,
            FOVEATE_SIZE,
            FOVEATE_SIZE,
            cgh,
            cgw,
        );

        Foveation {
            grid_fine,
            legal_tile_fine,
            grid_valid_fine,
            grid_coarse,
            gc_valid,
            legal_tile_coarse,
            fine_coarse,
            origin_y,
            origin_x,
            fine_h: FOVEATE_SIZE,
            fine_w: FOVEATE_SIZE,
        }
    }

    fn trunk_forward(&self, o: &Obs) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let fov = Self::foveate(o, self.foveate);

        let gc_map = self.grid_coarse_net.forward(&fov.grid_coarse, self.amp);
        let gc_valid_b = fov.gc_valid.unsqueeze(1);
        let gc_map = &gc_map * &gc_valid_b;
        let gc_pool = gc_map.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            / gc_valid_b.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float).clamp_min(1.0);

        let gf_map = self.grid_fine_net.forward(&fov.grid_fine, self.amp);
        let gf_valid_b = fov.grid_valid_fine.unsqueeze(1);
        let gf_map = &gf_map * &gf_valid_b;
        let gf_pool = gf_map.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            / gf_valid_b.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float).clamp_min(1.0);

        let mut p = self.player_in.forward(&o.players); // (B, S, PC)
        let key_pad_bias = (&o.pmask - 1.0).unsqueeze(1).unsqueeze(1) * (-MASKED_NEG); // (B,1,1,S)
        for layer in &self.tf_layers {
            p = layer.forward(&p, &key_pad_bias);
        }
        let m = o.pmask.unsqueeze(-1);
        let p_pool = (&p * &m).sum_dim_intlist(1i64, false, Kind::Float) / m.sum_dim_intlist(1i64, false, Kind::Float).clamp_min(1.0);

        let l_pool = self.local_net.forward(&o.local, self.amp);
        let cat = Tensor::cat(&[&gc_pool, &gf_pool, &p_pool, &l_pool, &o.scalars], -1);
        let h = self.trunk1.forward(&cat).silu();
        let h = self.trunk2.forward(&h).silu();
        (h, gc_map, gf_map, p, fov.grid_coarse)
    }

    fn tile_head(head: &(nn::Conv2D, nn::Conv2D), map: &Tensor, h: &Tensor, amp: bool) -> Tensor {
        let (b, _, gh, gw) = map.size4().unwrap();
        let hb = h.unsqueeze(-1).unsqueeze(-1).expand([b, HIDDEN, gh, gw], false);
        let cat = Tensor::cat(&[map, &hb], 1);
        // 1x1 convs over the full grid (up to GW_MAX x GH_MAX cells, GC +
        // HIDDEN input channels) - real compute, not just a cheap
        // per-pixel lookup, so worth running under the same bf16 path as
        // the towers; output cast back to f32 before it becomes the
        // tile logits (see `--amp` doc on `PolicyNet::amp`).
        if amp {
            let cat_b = cat.to_kind(Kind::BFloat16);
            let mid = conv2d_bf16(&head.0, &cat_b, [1, 1], [0, 0]).silu();
            conv2d_bf16(&head.1, &mid, [1, 1], [0, 0]).flatten(1, -1).to_kind(Kind::Float)
        } else {
            head.1.forward(&head.0.forward(&cat).silu()).flatten(1, -1)
        }
    }

    /// Full forward pass. Returns raw head tensors; callers combine with
    /// masks (see `act`/`evaluate`).
    fn forward(&self, o: &Obs) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (h, gc_map, gf_map, p, coarse_grid) = self.trunk_forward(o);
        let act_logits = self.head_action.forward(&h) + (&o.legal_actions - 1.0) * (-MASKED_NEG);
        let q = self.head_player_q.forward(&h); // (B, PC)
        let player_logits = q.unsqueeze(1).matmul(&p.transpose(-2, -1)).squeeze_dim(1); // (B, S)
        let tile_coarse = Self::tile_head(&self.head_tile_coarse, &gc_map, &h, self.amp);
        let tile_fine = Self::tile_head(&self.head_tile_fine, &gf_map, &h, self.amp);
        let build = self.head_build.forward(&h) + (&o.legal_build - 1.0) * (-MASKED_NEG);
        let nuke = self.head_nuke.forward(&h) + (&o.legal_nuke - 1.0) * (-MASKED_NEG);
        let quantity = self.head_quantity.forward(&h);
        let value = self.head_value.forward(&h).squeeze_dim(-1);
        (act_logits, player_logits, tile_coarse, tile_fine, build, nuke, quantity, value, p, coarse_grid)
    }

    fn coarse_dims(coarse_grid: &Tensor) -> (i64, i64) {
        let s = coarse_grid.size();
        (s[2], s[3])
    }

    /// `_coarse_logits_for_action`: coarse-mask (legal x land/water content
    /// pruning, v1 no-op since content is unknown - see foveate) blended
    /// with the fine->coarse "has legal fine cell" mask for REFINE_TILE
    /// actions. `fine_coarse` is `Foveation::fine_coarse` - already
    /// accounts for whether foveation is on (see `foveate`'s doc), so
    /// this function itself doesn't need to branch on it.
    fn coarse_logits_for_action(
        &self,
        tile_coarse: &Tensor,
        legal_tile_coarse: &Tensor,
        gc_valid: &Tensor,
        fine_coarse: &Tensor,
        action: &Tensor,
    ) -> Tensor {
        let base = gc_valid * legal_tile_coarse;
        let refine_action = action_table(REFINE_TILE, self.device).index_select(0, action);
        let has_fine =
            fine_coarse.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float).gt(0.0).to_kind(Kind::Float);
        let use_fine = (refine_action * has_fine).unsqueeze(-1).unsqueeze(-1);
        let one_minus_use_fine: Tensor = use_fine.neg() + 1.0;
        let mask: Tensor = fine_coarse * &use_fine + &base * one_minus_use_fine;
        tile_coarse + (mask.flatten(1, -1) - 1.0) * (-MASKED_NEG)
    }

    /// Legacy (`fine_origin` always 0) path: coarse cell containing >=1
    /// legal fine cell, computed via 2x2 max-pool of `legal_tile` over
    /// the WHOLE map instead of the scatter the general cropped-origin
    /// case needs (see `fine_to_coarse_mask_cropped`). Kept byte-for-byte
    /// as it always was so the `!foveate` path is provably unchanged.
    fn fine_to_coarse_mask(o: &Obs) -> Tensor {
        o.legal_tile.unsqueeze(1).max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true).squeeze_dim(1)
    }

    /// Legacy (`fine_origin` always 0) path: mask fine cells whose parent
    /// (gy/2, gx/2) equals the sampled coarse cell, intersected with
    /// legality; falls back to legal, then to all-valid. Kept
    /// byte-for-byte as it always was - see `fine_logits_for_coarse_cropped`
    /// for the real-crop equivalent.
    fn fine_logits_for_coarse(tile_fine: &Tensor, o: &Obs, coarse: &Tensor, cgw: i64) -> Tensor {
        let (b, gh, gw) = o.legal_tile.size3().unwrap();
        let dev = tile_fine.device();
        let cy = coarse.divide_scalar_mode(cgw, "floor");
        let cx = coarse.remainder(cgw);
        let yy = Tensor::arange(gh, (Kind::Int64, dev)).view([1, gh, 1]);
        let xx = Tensor::arange(gw, (Kind::Int64, dev)).view([1, 1, gw]);
        let gy = &yy / 2;
        let gx = &xx / 2;
        let mask = gy
            .eq_tensor(&cy.view([b, 1, 1]))
            .logical_and(&gx.eq_tensor(&cx.view([b, 1, 1])))
            .to_kind(Kind::Float);
        let mask = mask * &o.legal_tile * &o.grid_valid;
        let mask_sum = mask.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float);
        let fallback = &o.legal_tile * &o.grid_valid;
        let fb_sum = fallback.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float);
        let has_fb = fb_sum.gt(0.0).to_kind(Kind::Float).view([b, 1, 1]);
        let one_minus_has_fb: Tensor = has_fb.neg() + 1.0;
        let fallback2: Tensor = &fallback * &has_fb + &o.grid_valid * one_minus_has_fb;
        let has_mask = mask_sum.gt(0.0).to_kind(Kind::Float).view([b, 1, 1]);
        let one_minus_has_mask: Tensor = has_mask.neg() + 1.0;
        let mask: Tensor = &mask * &has_mask + &fallback2 * one_minus_has_mask;
        tile_fine + (mask.flatten(1, -1) - 1.0) * (-MASKED_NEG)
    }

    /// Dispatches to the legacy or real-crop coordinate math based on
    /// `self.foveate` - kept as thin `self`-branching wrappers so
    /// `act`/`evaluate` don't repeat the branch at every call site.
    fn fine_logits_for_coarse_any(&self, tile_fine: &Tensor, o: &Obs, fov: &Foveation, coarse: &Tensor, cgw: i64) -> Tensor {
        if self.foveate {
            fine_logits_for_coarse_cropped(
                tile_fine,
                &fov.legal_tile_fine,
                &fov.grid_valid_fine,
                &fov.origin_y,
                &fov.origin_x,
                fov.fine_h,
                fov.fine_w,
                coarse,
                cgw,
            )
        } else {
            Self::fine_logits_for_coarse(tile_fine, o, coarse, cgw)
        }
    }

    fn fine_local_to_global_any(&self, local: &Tensor, fov: &Foveation) -> Tensor {
        if self.foveate {
            fine_local_to_global_cropped(local, fov.fine_w, &fov.origin_y, &fov.origin_x)
        } else {
            fine_local_to_global(local, fov.fine_w)
        }
    }

    fn global_to_fine_local_any(&self, region: &Tensor, fov: &Foveation) -> Tensor {
        if self.foveate {
            global_to_fine_local_cropped(region, fov.fine_h, fov.fine_w, &fov.origin_y, &fov.origin_x)
        } else {
            global_to_fine_local(region, fov.fine_w)
        }
    }

    /// Batched sampling (mirrors `Policy.act`). `greedy=true` takes the
    /// argmax of every head. Returns (action, player_slot, tile_region,
    /// build_type, nuke_type, quantity_frac, logp, value); unused fields
    /// per-sample are left at their sampled/garbage value - callers must
    /// gate on the corresponding `needs_*`/`is_*` mask (see `vecenv`'s
    /// choice extraction) exactly like Python's `act()` dict-building loop.
    pub fn act(
        &self,
        o: &Obs,
        greedy: bool,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (act_logits, player_logits_raw, tile_coarse, tile_fine, build, nuke, quantity, value, _p, coarse_grid) =
            self.forward(o);
        let (a, mut logp) = categorical_sample(&act_logits, greedy);

        let pmask = o
            .legal_ptarget
            .gather(1, &a.view([-1, 1, 1]).expand([-1, 1, MAX_SLOTS as i64], false), false)
            .squeeze_dim(1);
        let player_logits = &player_logits_raw + (&pmask - 1.0) * (-MASKED_NEG);
        let (player, player_lp) = categorical_sample(&player_logits, greedy);
        let (build_s, build_lp) = categorical_sample(&build, greedy);
        let (nuke_s, nuke_lp) = categorical_sample(&nuke, greedy);

        let (qa, qb) = quantity_ab(&quantity);
        let q =
            if greedy { &qa / (&qa + &qb) } else { sample_beta_host(&qa, &qb) }.clamp(1e-4, 1.0 - 1e-4);
        let q_lp = beta_log_prob(&q, &qa, &qb);

        let dev = self.device;
        let needs_p = action_table(NEEDS_PLAYER, dev).index_select(0, &a);
        let needs_t = action_table(NEEDS_TILE, dev).index_select(0, &a);
        let needs_q = action_table(NEEDS_QUANTITY, dev).index_select(0, &a);
        let is_build =
            a.eq(ACTIONS.iter().position(|&x| x == "build").unwrap() as i64).to_kind(Kind::Float);
        let is_nuke =
            a.eq(ACTIONS.iter().position(|&x| x == "launch_nuke").unwrap() as i64).to_kind(Kind::Float);

        logp = logp + &needs_p * &player_lp;

        let (_cgh, cgw) = Self::coarse_dims(&coarse_grid);
        // Recomputes the crop (deterministic given `o.grid`, so this
        // matches what `forward()` used internally bit-for-bit) - only
        // needed here for the coordinate-translation fields (origin,
        // legal/valid masks), not the actual grid tensors already
        // consumed above.
        let fov = Self::foveate(o, self.foveate);
        let coarse_logits =
            self.coarse_logits_for_action(&tile_coarse, &fov.legal_tile_coarse, &fov.gc_valid, &fov.fine_coarse, &a);
        let (coarse, _coarse_lp_sampled) = categorical_sample(&coarse_logits, greedy);
        let fine_logits = self.fine_logits_for_coarse_any(&tile_fine, o, &fov, &coarse, cgw);
        let (fine, fine_lp) = categorical_sample(&fine_logits, greedy);

        let refine_bool = action_table(REFINE_TILE, dev).index_select(0, &a).to_kind(Kind::Bool);
        let fine_global = self.fine_local_to_global_any(&fine, &fov);
        let coarse_global = coarse_local_to_global(&coarse, cgw);
        let eff_coarse_local = global_to_coarse_local(&fine_global, cgw).where_self(&refine_bool, &coarse);
        let coarse_lp = categorical_logp(&coarse_logits, &eff_coarse_local);
        let tile_lp = (&coarse_lp + &fine_lp).where_self(&refine_bool, &coarse_lp);
        let tile_region = fine_global.where_self(&refine_bool, &coarse_global);

        logp = logp + &needs_t * &tile_lp;
        logp = logp + &is_build * &build_lp;
        logp = logp + &is_nuke * &nuke_lp;
        logp = logp + &needs_q * &q_lp;

        (a, player, tile_region, build_s, nuke_s, q, logp, value)
    }

    /// Batched logprob/entropy/value for PPO updates (mirrors
    /// `Policy.evaluate`). Every sub-head's contribution is computed over
    /// the FULL batch and zeroed via the "used" mask instead of Python's
    /// boolean-subset indexing (equivalent result, simpler on tch, and the
    /// extra compute is harmless for throughput testing).
    pub fn evaluate(&self, o: &Obs, c: &ChoiceBatch) -> (Tensor, Tensor, Tensor, Tensor) {
        let (act_logits, player_logits_raw, tile_coarse, tile_fine, build, nuke, quantity, value, _p, coarse_grid) =
            self.forward(o);
        let mut logp = categorical_logp(&act_logits, &c.action);
        let mut ent = categorical_entropy(&act_logits);

        let action_c = c.action.clamp(0, N_ACTIONS - 1);
        let pmask = o
            .legal_ptarget
            .gather(1, &action_c.view([-1, 1, 1]).expand([-1, 1, MAX_SLOTS as i64], false), false)
            .squeeze_dim(1);
        let player_logits = &player_logits_raw + (&pmask - 1.0) * (-MASKED_NEG);
        let p_used = c.player_slot.ge(0).to_kind(Kind::Float);
        let ps_c = c.player_slot.clamp(0, MAX_SLOTS as i64 - 1);
        logp = logp + &p_used * categorical_logp(&player_logits, &ps_c);
        ent = ent + &p_used * categorical_entropy(&player_logits);

        let (_gh, cgw) = Self::coarse_dims(&coarse_grid);
        // See `act`'s identical comment: recomputes the (deterministic)
        // crop for the coordinate-translation fields.
        let fov = Self::foveate(o, self.foveate);
        let t_used = c.tile_region.ge(0).to_kind(Kind::Float);
        let tr_c = c.tile_region.clamp(0, i64::MAX / 2);
        let coarse_target = global_to_coarse_local(&tr_c, cgw);
        let coarse_logits =
            self.coarse_logits_for_action(&tile_coarse, &fov.legal_tile_coarse, &fov.gc_valid, &fov.fine_coarse, &action_c);
        let coarse_target_c = coarse_target.clamp(0, cgw * Self::coarse_dims(&coarse_grid).0 - 1);
        logp = logp + &t_used * categorical_logp(&coarse_logits, &coarse_target_c);
        ent = ent + &t_used * categorical_entropy(&coarse_logits);

        let refine = action_table(REFINE_TILE, self.device).index_select(0, &action_c) * &t_used;
        let fine_target = self.global_to_fine_local_any(&tr_c, &fov);
        let fine_target_c = fine_target.clamp(0, fov.fine_h * fov.fine_w - 1);
        let fine_logits = self.fine_logits_for_coarse_any(&tile_fine, o, &fov, &coarse_target_c, cgw);
        logp = logp + &refine * categorical_logp(&fine_logits, &fine_target_c);
        ent = ent + &refine * categorical_entropy(&fine_logits);

        let b_used = c.build_type.ge(0).to_kind(Kind::Float);
        let bt_c = c.build_type.clamp(0, N_BUILD - 1);
        logp = logp + &b_used * categorical_logp(&build, &bt_c);
        ent = ent + &b_used * categorical_entropy(&build);

        let n_used = c.nuke_type.ge(0).to_kind(Kind::Float);
        let nt_c = c.nuke_type.clamp(0, N_NUKE - 1);
        logp = logp + &n_used * categorical_logp(&nuke, &nt_c);
        ent = ent + &n_used * categorical_entropy(&nuke);

        let (qa, qb) = quantity_ab(&quantity);
        let q_used = c.quantity_frac.ge(0.0).to_kind(Kind::Float);
        let q_target = c.quantity_frac.clamp(1e-4, 1.0 - 1e-4);
        let q_lp = beta_log_prob(&q_target, &qa, &qb);
        logp = logp + &q_used * &q_lp;
        let ent_q = &q_used * beta_entropy(&qa, &qb);

        (logp, ent, ent_q, value)
    }

    pub fn value_only(&self, o: &Obs) -> Tensor {
        let (h, _, _, _, _) = self.trunk_forward(o);
        self.head_value.forward(&h).squeeze_dim(-1)
    }
}

/// Coarse local index -> global index, in the SAME (fine-scale) global
/// coordinate system `fine_local_to_global` uses: coarse cells are 2x2
/// fine cells, so both coordinates are doubled (`_coarse_local_to_global`).
fn coarse_local_to_global(local: &Tensor, cgw: i64) -> Tensor {
    let cy = local.divide_scalar_mode(cgw, "floor");
    let cx = local.remainder(cgw);
    (cy * 2) * GW_MAX + cx * 2
}

fn fine_local_to_global(local: &Tensor, gw: i64) -> Tensor {
    (local.divide_scalar_mode(gw, "floor")) * GW_MAX + local.remainder(gw)
}

/// Global (fine-scale) index -> coarse local index: divide both
/// coordinates by 2 before re-flattening at coarse stride
/// (`_global_to_coarse_local`).
fn global_to_coarse_local(region: &Tensor, cgw: i64) -> Tensor {
    let r = region.clamp_min(0);
    let gy = r.divide_scalar_mode(GW_MAX, "floor");
    let gx = r.remainder(GW_MAX);
    (gy.divide_scalar_mode(2, "floor")) * cgw + gx.divide_scalar_mode(2, "floor")
}

fn global_to_fine_local(region: &Tensor, gw: i64) -> Tensor {
    let r = region.clamp_min(0);
    (r.divide_scalar_mode(GW_MAX, "floor")) * gw + r.remainder(GW_MAX)
}

/// Host-side Beta sampling (act() runs under no_grad; a tiny (B,) round
/// trip through the CPU rand crate is cheaper than implementing a
/// differentiable-agnostic Gamma sampler in tch, and this path never
/// needs gradients).
fn sample_beta_host(a: &Tensor, b: &Tensor) -> Tensor {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Beta, Distribution};
    let dev = a.device();
    let a_cpu = a.to_device(Device::Cpu).to_kind(Kind::Float);
    let b_cpu = b.to_device(Device::Cpu).to_kind(Kind::Float);
    let n = a_cpu.numel();
    let av: Vec<f32> = a_cpu.reshape([-1]).try_into().unwrap();
    let bv: Vec<f32> = b_cpu.reshape([-1]).try_into().unwrap();
    let mut rng = SmallRng::from_entropy();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let d = Beta::new(av[i].max(1e-3) as f64, bv[i].max(1e-3) as f64).unwrap();
        out.push(d.sample(&mut rng) as f32);
    }
    Tensor::from_slice(&out).view(a.size().as_slice()).to_device(dev)
}

/// `--foveate` crop origin: the top-left (row, col) of a `ch`x`cw` window
/// centered on the weighted centroid of non-zero cells in `mask` (B, h, w),
/// clamped to stay fully inside the (h, w) grid and snapped down to an even
/// coordinate (so the crop's fine cells stay aligned to the coarse grid's
/// 2x2 blocks - every downstream coarse<->fine mapping assumes that).
/// Falls back to the grid's center when a sample's `mask` is all-zero
/// (e.g. spawn phase, before the agent owns any tile), exactly mirroring
/// `ofcore::feat::local_crop`'s identical fallback for the unrelated
/// LocalNet branch. Fully vectorized (no host round trip / per-sample
/// loop), so this stays cheap at real batch sizes on GPU.
fn crop_origin(mask: &Tensor, h: i64, w: i64, ch: i64, cw: i64) -> (Tensor, Tensor) {
    let dev = mask.device();
    let yy = Tensor::arange(h, (Kind::Float, dev)).view([1, h, 1]);
    let xx = Tensor::arange(w, (Kind::Float, dev)).view([1, 1, w]);

    let n = mask.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float); // (B,)
    let has_any = n.gt(0.0).to_kind(Kind::Float);
    let n_safe = n.clamp_min(1.0);
    let sum_y = (&yy * mask).sum_dim_intlist([1, 2].as_slice(), false, Kind::Float);
    let sum_x = (&xx * mask).sum_dim_intlist([1, 2].as_slice(), false, Kind::Float);
    let centroid_y = &sum_y / &n_safe;
    let centroid_x = &sum_x / &n_safe;
    let no_owned: Tensor = has_any.neg() + 1.0;
    let cy = (&has_any * &centroid_y + &no_owned * ((h - 1) as f64 / 2.0)).round().to_kind(Kind::Int64);
    let cx = (&has_any * &centroid_x + &no_owned * ((w - 1) as f64 / 2.0)).round().to_kind(Kind::Int64);

    let h_half = ch / 2;
    let w_half = cw / 2;
    let oy = (cy - h_half).clamp(0, h - ch);
    let ox = (cx - w_half).clamp(0, w - cw);
    let origin_y: Tensor = &oy - oy.remainder(2);
    let origin_x: Tensor = &ox - ox.remainder(2);
    (origin_y, origin_x)
}

/// Gathers a `crop_h`x`crop_w` window from `src` (B, C, H, W) at each
/// sample's `origin_y`/`origin_x` (absolute top-left, both (B,) i64
/// tensors), then zero-pads it up to `out_h`x`out_w` if the crop is
/// smaller (only possible when the source grid itself is smaller than the
/// requested crop, e.g. a tiny map with `FOVEATE_SIZE` clamped down in
/// `PolicyNet::foveate`). No host round trip: the whole batch's crop is a
/// single vectorized `gather` call.
fn crop_and_pad(
    src: &Tensor,
    origin_y: &Tensor,
    origin_x: &Tensor,
    crop_h: i64,
    crop_w: i64,
    out_h: i64,
    out_w: i64,
) -> Tensor {
    let (b, c, src_h, src_w) = src.size4().unwrap();
    let dev = src.device();

    let target_y = Tensor::arange(crop_h, (Kind::Int64, dev)).view([1, crop_h, 1]);
    let target_x = Tensor::arange(crop_w, (Kind::Int64, dev)).view([1, 1, crop_w]);
    let abs_y = (&target_y + origin_y.view([-1, 1, 1])).clamp(0, src_h - 1);
    let abs_x = (&target_x + origin_x.view([-1, 1, 1])).clamp(0, src_w - 1);
    let flat_idx = (abs_y * src_w + abs_x).view([b, 1, crop_h * crop_w]).expand([b, c, crop_h * crop_w], false);

    let src_flat = src.flatten(2, -1); // (B, C, src_h * src_w)
    let cropped = src_flat.gather(-1, &flat_idx, false).view([b, c, crop_h, crop_w]);

    if crop_h == out_h && crop_w == out_w {
        cropped
    } else {
        // constant_pad_nd pads last-dim-first: [w_before, w_after, h_before, h_after].
        cropped.constant_pad_nd([0, out_w - crop_w, 0, out_h - crop_h])
    }
}

/// Inverse of `crop_and_pad`'s gather: places a (B, lh, lw) local-space
/// tensor into a (B, oh, ow) full-space tensor at each sample's
/// `origin_y`/`origin_x` (already in the *target* `oh`/`ow` coordinate
/// system, e.g. coarse-scale origin = fine-scale origin / 2), zero outside
/// the placed window. Implemented as a gather walking the full output grid
/// back to local coordinates (masking out-of-window cells) rather than a
/// scatter, so it stays a single vectorized op with no per-sample loop.
fn place_crop(
    local: &Tensor,
    origin_y: &Tensor,
    origin_x: &Tensor,
    lh: i64,
    lw: i64,
    oh: i64,
    ow: i64,
) -> Tensor {
    let dev = local.device();
    let (b, _, _) = local.size3().unwrap();
    let oy = Tensor::arange(oh, (Kind::Int64, dev)).view([1, oh, 1]);
    let ox = Tensor::arange(ow, (Kind::Int64, dev)).view([1, 1, ow]);
    let ly = &oy - origin_y.view([-1, 1, 1]);
    let lx = &ox - origin_x.view([-1, 1, 1]);
    let in_bounds = ly
        .ge(0)
        .logical_and(&ly.lt(lh))
        .logical_and(&lx.ge(0))
        .logical_and(&lx.lt(lw))
        .to_kind(Kind::Float); // (B, oh, ow)

    let ly_c = ly.clamp(0, lh - 1);
    let lx_c = lx.clamp(0, lw - 1);
    let flat_idx = (ly_c * lw + lx_c).view([b, oh * ow]); // (B, oh*ow)
    let local_flat = local.flatten(1, -1); // (B, lh*lw)
    let gathered = local_flat.gather(1, &flat_idx, false).view([b, oh, ow]);
    gathered * in_bounds
}

/// `--foveate` real-crop path: coarse cell (in FULL-map coarse space)
/// containing >=1 legal+valid fine cell within the crop window. `_fine`
/// args are already the crop+pad'd (FOVEATE_SIZE-sized) tensors from
/// `PolicyNet::foveate`. Computed as a local 2x2 max-pool (matching the
/// legacy `fine_to_coarse_mask`'s pooling exactly, just over the crop
/// instead of the whole map) then placed into full coarse-map space via
/// `place_crop` - the real-crop equivalent of the legacy path's implicit
/// "coarse == fine/2 everywhere" identity.
fn fine_to_coarse_mask_cropped(
    legal_tile_fine: &Tensor,
    grid_valid_fine: &Tensor,
    origin_y: &Tensor,
    origin_x: &Tensor,
    fine_h: i64,
    fine_w: i64,
    cgh: i64,
    cgw: i64,
) -> Tensor {
    let local_legal_valid = legal_tile_fine * grid_valid_fine; // (B, fine_h, fine_w)
    let local_coarse = local_legal_valid
        .unsqueeze(1)
        .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
        .squeeze_dim(1); // (B, fine_h/2, fine_w/2)
    let origin_cy = origin_y.divide_scalar_mode(2, "floor");
    let origin_cx = origin_x.divide_scalar_mode(2, "floor");
    place_crop(&local_coarse, &origin_cy, &origin_cx, fine_h / 2, fine_w / 2, cgh, cgw)
}

/// `--foveate` real-crop path: mask fine cells (in the crop's LOCAL
/// coordinate system) whose absolute parent coarse cell equals the sampled
/// `coarse` index (a full-map coarse-space index, same as the legacy
/// path's), intersected with legality/validity; falls back to legal, then
/// to all-valid, exactly mirroring `fine_logits_for_coarse`'s fallback
/// chain. The only difference from the legacy version is translating the
/// sampled coarse cell's absolute fine-scale coordinates into the crop's
/// local frame by subtracting `origin_y`/`origin_x` before comparing
/// against the local `yy`/`xx` meshgrid.
fn fine_logits_for_coarse_cropped(
    tile_fine: &Tensor,
    legal_tile_fine: &Tensor,
    grid_valid_fine: &Tensor,
    origin_y: &Tensor,
    origin_x: &Tensor,
    fine_h: i64,
    fine_w: i64,
    coarse: &Tensor,
    cgw: i64,
) -> Tensor {
    // `tile_fine` is already flat (B, fine_h*fine_w) - see `tile_head` -
    // so `b` comes from one of the still-3D crop tensors instead.
    let (b, _, _) = legal_tile_fine.size3().unwrap();
    let dev = tile_fine.device();

    // Absolute fine-scale top-left of the sampled coarse cell's parent
    // block, translated into the crop's local frame. `coarse`/`origin_*`
    // are both (B,) here - no `.view` needed until the broadcast against
    // the (1, fine_h, 1)/(1, 1, fine_w) meshgrids below.
    let local_top_y: Tensor = coarse.divide_scalar_mode(cgw, "floor") * 2 - origin_y;
    let local_top_x: Tensor = coarse.remainder(cgw) * 2 - origin_x;

    let yy = Tensor::arange(fine_h, (Kind::Int64, dev)).view([1, fine_h, 1]);
    let xx = Tensor::arange(fine_w, (Kind::Int64, dev)).view([1, 1, fine_w]);

    let mask = yy
        .ge_tensor(&local_top_y.view([b, 1, 1]))
        .logical_and(&yy.lt_tensor(&(&local_top_y + 2).view([b, 1, 1])))
        .logical_and(&xx.ge_tensor(&local_top_x.view([b, 1, 1])))
        .logical_and(&xx.lt_tensor(&(&local_top_x + 2).view([b, 1, 1])))
        .to_kind(Kind::Float);

    let mask = mask * legal_tile_fine * grid_valid_fine;
    let mask_sum = mask.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float);
    let fallback = legal_tile_fine * grid_valid_fine;
    let fb_sum = fallback.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float);
    let has_fb = fb_sum.gt(0.0).to_kind(Kind::Float).view([b, 1, 1]);
    let one_minus_has_fb: Tensor = has_fb.neg() + 1.0;
    let fallback2: Tensor = &fallback * &has_fb + grid_valid_fine * one_minus_has_fb;
    let has_mask = mask_sum.gt(0.0).to_kind(Kind::Float).view([b, 1, 1]);
    let one_minus_has_mask: Tensor = has_mask.neg() + 1.0;
    let mask: Tensor = &mask * &has_mask + &fallback2 * one_minus_has_mask;

    tile_fine + (mask.flatten(1, -1) - 1.0) * (-MASKED_NEG)
}

/// `--foveate` real-crop path: local fine index within the crop -> global
/// (fine-scale, GW_MAX-strided) index, by first adding back the crop's
/// absolute origin then flattening at the *global* map width (see the
/// legacy `fine_local_to_global`, whose `local` is already global-frame
/// since it has no crop to translate out of).
fn fine_local_to_global_cropped(local: &Tensor, fine_w: i64, origin_y: &Tensor, origin_x: &Tensor) -> Tensor {
    // `local`/`origin_*` are all (B,) here (see the legacy `fine_local_to_global`,
    // which is the same shape contract without a crop to translate out of).
    let ly = local.divide_scalar_mode(fine_w, "floor");
    let lx = local.remainder(fine_w);
    (&ly + origin_y) * GW_MAX + (&lx + origin_x)
}

/// `--foveate` real-crop path: global (fine-scale) index -> local fine
/// index within the crop, or -1 if the global cell falls outside the
/// crop window (this happens whenever the target tile isn't within the
/// currently-foveated region - callers must already be gating on
/// `REFINE_TILE`/coarse-fallback exactly like the legacy path, so a -1
/// here just means "irrelevant for this action", not an error).
fn global_to_fine_local_cropped(
    region: &Tensor,
    fine_h: i64,
    fine_w: i64,
    origin_y: &Tensor,
    origin_x: &Tensor,
) -> Tensor {
    // `region`/`origin_*` are all (B,) here (see `fine_local_to_global_cropped`).
    let r = region.clamp_min(0);
    let gy = r.divide_scalar_mode(GW_MAX, "floor");
    let gx = r.remainder(GW_MAX);

    let ly = gy - origin_y;
    let lx = gx - origin_x;

    let mask = ly.ge(0).logical_and(&ly.lt(fine_h)).logical_and(&lx.ge(0)).logical_and(&lx.lt(fine_w));

    let local = &ly * fine_w + &lx;
    local.where_self(&mask, &(local.zeros_like() - 1))
}

#[cfg(test)]
mod tests {
    //! Fast, small-scale (tiny synthetic grids, not a real map) forward/
    //! backward correctness tests for the `--amp`/`--foveate`/`--gc`/
    //! `--blocks` code paths - deliberately bypassing the Node engine
    //! bridge entirely so these run in milliseconds regardless of engine
    //! availability. This is the practical way to smoke-test `--amp` on
    //! CPU at all: a real-map-scale (up to GH_MAX x GW_MAX, GC=256,
    //! BLOCKS=4) forward pass under manually-cast bf16 was measured to be
    //! *dramatically* slower than f32 on this CPU (no accelerated CPU
    //! bf16 GEMM/conv kernel in this libtorch build - still correct, just
    //! not practically smoke-testable end-to-end without a GPU; see
    //! DEVLOG) - a single tiny-grid batch keeps the same code path
    //! exercised while keeping wall-clock trivial.
    use super::*;

    fn synthetic_obs(device: Device, b: i64, gh: i64, gw: i64) -> Obs {
        let ms = MAX_SLOTS;
        let na = N_ACTIONS;
        let opts = (Kind::Float, device);
        Obs {
            grid: Tensor::rand([b, C_GRID, gh, gw], opts),
            grid_valid: Tensor::ones([b, gh, gw], opts),
            legal_tile: Tensor::ones([b, gh, gw], opts),
            players: Tensor::rand([b, ms, P_FEAT], opts),
            pmask: Tensor::ones([b, ms], opts),
            local: Tensor::rand([b, N_LOCAL, LOCAL, LOCAL], opts),
            scalars: Tensor::rand([b, N_SCALARS], opts),
            legal_actions: Tensor::ones([b, na], opts),
            legal_ptarget: Tensor::ones([b, na, ms], opts),
            legal_build: Tensor::ones([b, N_BUILD], opts),
            legal_nuke: Tensor::ones([b, N_NUKE], opts),
        }
    }

    fn assert_finite(t: &Tensor, what: &str) {
        let all_finite = t.isfinite().all().double_value(&[]);
        assert!(all_finite != 0.0, "{what} has non-finite values: {t:?}");
    }

    /// Exercises `act()` (no_grad) + `evaluate()` + `backward()` for a
    /// given `PolicyNet` config, asserting every returned tensor and the
    /// summed loss stay finite. Shared by the amp/foveate/model-size
    /// tests below so each only has to say what's different about it.
    fn check_policy_finite(policy: &PolicyNet, o: &Obs) {
        let (a, player, tile, build, nuke, qty, logp, value) = tch::no_grad(|| policy.act(o, false));
        for (name, t) in [
            ("act.logp", &logp),
            ("act.value", &value),
            ("act.qty", &qty),
        ] {
            assert_finite(t, name);
        }

        let choice = ChoiceBatch {
            action: a,
            player_slot: player,
            tile_region: tile,
            build_type: build,
            nuke_type: nuke,
            quantity_frac: qty,
        };
        let (logp2, ent, ent_q, value2) = policy.evaluate(o, &choice);
        assert_finite(&logp2, "evaluate.logp");
        assert_finite(&ent, "evaluate.ent");
        assert_finite(&ent_q, "evaluate.ent_q");
        assert_finite(&value2, "evaluate.value");

        let loss = logp2.mean(Kind::Float) + ent.mean(Kind::Float) + ent_q.mean(Kind::Float)
            + value2.mean(Kind::Float);
        loss.backward();
        let loss_v = loss.double_value(&[]);
        assert!(loss_v.is_finite(), "loss not finite: {loss_v}");
    }

    #[test]
    fn amp_and_f32_paths_finite() {
        tch::manual_seed(0);
        for &amp in &[false, true] {
            let vs = nn::VarStore::new(Device::Cpu);
            let policy = PolicyNet::new(&vs.root(), amp, false);
            let o = synthetic_obs(Device::Cpu, 2, 6, 6);
            check_policy_finite(&policy, &o);
        }
    }

    /// `--foveate` on a grid bigger than `FOVEATE_SIZE` in both dims, so
    /// the crop is a genuine strict subset of the map (exercises the
    /// non-degenerate `crop_and_pad`/`place_crop` path, not just the
    /// "crop == whole map" edge case a tiny grid would hit).
    #[test]
    fn foveate_crop_finite() {
        tch::manual_seed(1);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true);
        let o = synthetic_obs(Device::Cpu, 2, FOVEATE_SIZE * 2, FOVEATE_SIZE * 2 + 4);
        check_policy_finite(&policy, &o);
    }

    /// `--foveate` on a grid smaller than `FOVEATE_SIZE` in both dims,
    /// exercising the zero-pad branch of `crop_and_pad` (crop == whole
    /// map, padded out to the fixed window size).
    #[test]
    fn foveate_crop_smaller_than_window_finite() {
        tch::manual_seed(2);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true);
        let o = synthetic_obs(Device::Cpu, 2, 6, 8);
        check_policy_finite(&policy, &o);
    }

    /// `--foveate` combined with `--amp`, since both mutate the same
    /// tile-head/tower forward paths and could interact badly.
    #[test]
    fn foveate_and_amp_finite() {
        tch::manual_seed(3);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), true, true);
        let o = synthetic_obs(Device::Cpu, 2, FOVEATE_SIZE * 2, FOVEATE_SIZE * 2);
        check_policy_finite(&policy, &o);
    }

    /// The real-crop coordinate math (`crop_origin`/`crop_and_pad`/
    /// `place_crop`/the `_cropped` helpers) against hand-computed
    /// expectations, independent of the full network - this is the
    /// highest-risk part of `--foveate`, so it gets a dedicated,
    /// non-random test instead of only relying on "the loss is finite".
    #[test]
    fn foveate_coordinate_math() {
        let dev = Device::Cpu;
        // Sample 0: owns a single tile at (10, 12) -> centroid (10, 12),
        // window size 4x4 -> origin clamped/snapped to (8, 10) (nearest
        // even <= centroid - half-window, within [0, h-ch]).
        // Sample 1: owns nothing -> falls back to grid center (h-1)/2,
        // (w-1)/2 = (9.5, 9.5) -> rounds to (10, 10) -> origin (8, 8).
        let (h, w, ch, cw) = (20, 20, 4, 4);
        let mut mask = vec![0.0f32; 2 * h as usize * w as usize];
        mask[(10 * w + 12) as usize] = 1.0;
        let mask_t = Tensor::from_slice(&mask).view([2, h, w]).to_device(dev);
        let (oy, ox) = crop_origin(&mask_t, h, w, ch, cw);
        let oy_v: Vec<i64> = oy.reshape([-1]).try_into().unwrap();
        let ox_v: Vec<i64> = ox.reshape([-1]).try_into().unwrap();
        assert_eq!(oy_v, vec![8, 8]);
        assert_eq!(ox_v, vec![10, 8]);

        // crop_and_pad: gather a 4x4 window at origin (8, 10) out of a
        // single-channel 20x20 iota grid, no padding needed (out size ==
        // crop size).
        let iota: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
        let src = Tensor::from_slice(&iota).view([1, 1, h, w]).to_device(dev);
        let oy1 = Tensor::from_slice(&[8i64]).to_device(dev);
        let ox1 = Tensor::from_slice(&[10i64]).to_device(dev);
        let cropped = crop_and_pad(&src, &oy1, &ox1, ch, cw, ch, cw);
        let cropped_v: Vec<f32> = cropped.reshape([-1]).try_into().unwrap();
        let expected: Vec<f32> =
            (0..ch).flat_map(|dy| (0..cw).map(move |dx| ((8 + dy) * w + 10 + dx) as f32)).collect();
        assert_eq!(cropped_v, expected);

        // place_crop is the inverse: placing that same 4x4 block back at
        // (8, 10) into a 20x20 zero canvas should reproduce the original
        // values there and zero everywhere else. place_crop takes a
        // (B, H, W) tensor (no channel dim - see its callers), so squeeze
        // the single channel out of `cropped` first.
        let placed = place_crop(&cropped.squeeze_dim(1), &oy1, &ox1, ch, cw, h, w);
        let placed_v: Vec<f32> = placed.reshape([-1]).try_into().unwrap();
        for y in 0..h {
            for x in 0..w {
                let expect = if (8..8 + ch).contains(&y) && (10..10 + cw).contains(&x) {
                    (y * w + x) as f32
                } else {
                    0.0
                };
                assert_eq!(placed_v[(y * w + x) as usize], expect, "mismatch at ({y}, {x})");
            }
        }

        // global_to_fine_local_cropped / fine_local_to_global_cropped
        // round-trip for a cell inside the crop, and the global->local
        // direction returns -1 for a cell outside it.
        let inside_global = Tensor::from_slice(&[(9i64) * GW_MAX + 11]).to_device(dev); // (9,11) is inside [8,12)x[10,14)
        let local = global_to_fine_local_cropped(&inside_global, ch, cw, &oy1, &ox1);
        let local_v: Vec<i64> = local.reshape([-1]).try_into().unwrap();
        assert_eq!(local_v, vec![(9 - 8) * cw + (11 - 10)]);
        let roundtrip = fine_local_to_global_cropped(&local, cw, &oy1, &ox1);
        let roundtrip_v: Vec<i64> = roundtrip.reshape([-1]).try_into().unwrap();
        let inside_global_v: Vec<i64> = inside_global.reshape([-1]).try_into().unwrap();
        assert_eq!(roundtrip_v, inside_global_v);

        let outside_global = Tensor::from_slice(&[0i64 * GW_MAX + 0]).to_device(dev); // (0,0) outside the crop
        let local_outside = global_to_fine_local_cropped(&outside_global, ch, cw, &oy1, &ox1);
        let local_outside_v: Vec<i64> = local_outside.reshape([-1]).try_into().unwrap();
        assert_eq!(local_outside_v, vec![-1]);
    }
}
