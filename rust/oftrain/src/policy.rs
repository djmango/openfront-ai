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

struct ResBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
}

impl ResBlock {
    fn new(p: &nn::Path, c: i64) -> Self {
        let cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        ResBlock { conv1: nn::conv2d(p / "conv1", c, c, 3, cfg), conv2: nn::conv2d(p / "conv2", c, c, 3, cfg) }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv1.forward(x).silu();
        (x + self.conv2.forward(&h)).silu()
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
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.stem.forward(x).silu();
        for b in &self.blocks {
            h = b.forward(&h);
        }
        h
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
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.c1.forward(x).silu();
        let h = self.c2.forward(&h).silu();
        let h = self.c3.forward(&h).silu();
        h.adaptive_avg_pool2d([1, 1]).flatten(1, -1)
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
}

impl PolicyNet {
    pub fn new(vs: &nn::Path) -> Self {
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
        }
    }

    /// Derives fine/coarse grid + validity + legal-tile tensors from the
    /// single full-res `grid`/`grid_valid`/`legal_tile` inputs, matching
    /// `Policy._ensure_foveated`'s legacy fallback (see module doc).
    fn foveate(o: &Obs) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let grid_fine = Tensor::cat(&[&o.grid, &o.grid_valid.unsqueeze(1)], 1);
        let grid_coarse =
            o.grid.avg_pool2d([2, 2], [2, 2], [0, 0], true, false, None::<i64>);
        let grid_coarse_valid = o
            .grid_valid
            .unsqueeze(1)
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
            .squeeze_dim(1);
        let legal_tile_coarse = o
            .legal_tile
            .unsqueeze(1)
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
            .squeeze_dim(1);
        let coarse_has_land = grid_coarse_valid.ones_like();
        let coarse_has_water = grid_coarse_valid.ones_like();
        (grid_fine, grid_coarse, grid_coarse_valid, legal_tile_coarse, coarse_has_land, coarse_has_water)
    }

    fn trunk_forward(&self, o: &Obs) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (grid_fine, grid_coarse, gc_valid, _legal_tile_coarse, _hl, _hw) = Self::foveate(o);

        let gc_map = self.grid_coarse_net.forward(&grid_coarse);
        let gc_valid_b = gc_valid.unsqueeze(1);
        let gc_map = &gc_map * &gc_valid_b;
        let gc_pool = gc_map.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            / gc_valid_b.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float).clamp_min(1.0);

        let gf_map = self.grid_fine_net.forward(&grid_fine);
        let gf_valid_b = o.grid_valid.unsqueeze(1);
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

        let l_pool = self.local_net.forward(&o.local);
        let cat = Tensor::cat(&[&gc_pool, &gf_pool, &p_pool, &l_pool, &o.scalars], -1);
        let h = self.trunk1.forward(&cat).silu();
        let h = self.trunk2.forward(&h).silu();
        (h, gc_map, gf_map, p, grid_coarse)
    }

    fn tile_head(head: &(nn::Conv2D, nn::Conv2D), map: &Tensor, h: &Tensor) -> Tensor {
        let (b, _, gh, gw) = map.size4().unwrap();
        let hb = h.unsqueeze(-1).unsqueeze(-1).expand([b, HIDDEN, gh, gw], false);
        let cat = Tensor::cat(&[map, &hb], 1);
        head.1.forward(&head.0.forward(&cat).silu()).flatten(1, -1)
    }

    /// Full forward pass. Returns raw head tensors; callers combine with
    /// masks (see `act`/`evaluate`).
    fn forward(&self, o: &Obs) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (h, gc_map, gf_map, p, coarse_grid) = self.trunk_forward(o);
        let act_logits = self.head_action.forward(&h) + (&o.legal_actions - 1.0) * (-MASKED_NEG);
        let q = self.head_player_q.forward(&h); // (B, PC)
        let player_logits = q.unsqueeze(1).matmul(&p.transpose(-2, -1)).squeeze_dim(1); // (B, S)
        let tile_coarse = Self::tile_head(&self.head_tile_coarse, &gc_map, &h);
        let tile_fine = Self::tile_head(&self.head_tile_fine, &gf_map, &h);
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
    /// actions.
    fn coarse_logits_for_action(
        &self,
        tile_coarse: &Tensor,
        o: &Obs,
        legal_tile_coarse: &Tensor,
        gc_valid: &Tensor,
        action: &Tensor,
    ) -> Tensor {
        let base = gc_valid * legal_tile_coarse;
        let refine_action = action_table(REFINE_TILE, self.device).index_select(0, action);
        let fine_coarse = Self::fine_to_coarse_mask(o);
        let has_fine =
            fine_coarse.flatten(1, -1).sum_dim_intlist(1i64, false, Kind::Float).gt(0.0).to_kind(Kind::Float);
        let use_fine = (refine_action * has_fine).unsqueeze(-1).unsqueeze(-1);
        let one_minus_use_fine: Tensor = use_fine.neg() + 1.0;
        let mask: Tensor = &fine_coarse * &use_fine + &base * one_minus_use_fine;
        tile_coarse + (mask.flatten(1, -1) - 1.0) * (-MASKED_NEG)
    }

    /// v1 (fine_origin always 0): coarse cell containing >=1 legal fine
    /// cell, computed via 2x2 max-pool of `legal_tile` instead of the
    /// scatter-add Python uses for the general cropped-origin case.
    fn fine_to_coarse_mask(o: &Obs) -> Tensor {
        o.legal_tile.unsqueeze(1).max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true).squeeze_dim(1)
    }

    /// `_fine_logits_for_coarse` (fine_origin=0): mask fine cells whose
    /// parent (gy/2, gx/2) equals the sampled coarse cell, intersected
    /// with legality; falls back to legal, then to all-valid.
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
        let gw = o.grid.size()[3];
        let (_gf, _gc, gc_valid, legal_tile_coarse, _hl, _hw) = Self::foveate(o);
        let coarse_logits = self.coarse_logits_for_action(&tile_coarse, o, &legal_tile_coarse, &gc_valid, &a);
        let (coarse, _coarse_lp_sampled) = categorical_sample(&coarse_logits, greedy);
        let fine_logits = Self::fine_logits_for_coarse(&tile_fine, o, &coarse, cgw);
        let (fine, fine_lp) = categorical_sample(&fine_logits, greedy);

        let refine_bool = action_table(REFINE_TILE, dev).index_select(0, &a).to_kind(Kind::Bool);
        let fine_global = fine_local_to_global(&fine, gw);
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
        let gw = o.grid.size()[3];
        let (_gf, _gc, gc_valid, legal_tile_coarse, _hl, _hw) = Self::foveate(o);
        let t_used = c.tile_region.ge(0).to_kind(Kind::Float);
        let tr_c = c.tile_region.clamp(0, i64::MAX / 2);
        let coarse_target = global_to_coarse_local(&tr_c, cgw);
        let coarse_logits =
            self.coarse_logits_for_action(&tile_coarse, o, &legal_tile_coarse, &gc_valid, &action_c);
        let coarse_target_c = coarse_target.clamp(0, cgw * Self::coarse_dims(&coarse_grid).0 - 1);
        logp = logp + &t_used * categorical_logp(&coarse_logits, &coarse_target_c);
        ent = ent + &t_used * categorical_entropy(&coarse_logits);

        let refine = action_table(REFINE_TILE, self.device).index_select(0, &action_c) * &t_used;
        let fine_target = global_to_fine_local(&tr_c, gw);
        let fine_target_c = fine_target.clamp(0, gw * o.grid.size()[2] - 1);
        let fine_logits = Self::fine_logits_for_coarse(&tile_fine, o, &coarse_target_c, cgw);
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
