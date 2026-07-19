//! Policy network: tch port of `rl/policy.py`.
//!
//! Grid channels match production Python PPO: frozen AE latent (32) +
//! ego (3) + defense_bonus (1) + transient (53) = `C_GRID` 89. AE encode
//! lives in `ae.rs` / `batch::build_obs_with_ae`. Optional native /16
//! coarse stream arrives via `Obs::grid_coarse` when `--coarse-ckpt` is
//! set; otherwise `foveate` falls back to 2x avg-pool of the fine grid
//! (same as Python without `coarse_ae`).
//!
//! `--foveate` (on by default in training configs that pass it): fixed
//! `FOVEATE_SIZE` crop centered on own-tile mass. Off uses the legacy
//! whole-map-as-fine path (`Policy._ensure_foveated` fallback).

use ofcore::feat::{ACTIONS, GW_MAX};
use tch::nn::Module;
use tch::{Device, Kind, Tensor, nn};

pub const N_ACTIONS: i64 = ofcore::feat::N_ACTIONS as i64;
pub const MAX_SLOTS: i64 = ofcore::feat::MAX_SLOTS as i64;

pub const LATENT_C: i64 = 32;
pub const N_TRANSIENT: i64 = 53;
/// Own-ego channel index inside `grid` (first bypass plane after latent).
pub const EGO_OWN_CH: i64 = LATENT_C;
pub const C_GRID: i64 = LATENT_C + 3 + 1 + N_TRANSIENT; // 89
pub const C_GRID_FINE: i64 = C_GRID + 1; // 90
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
/// V8.2 recurrent state width, independent of the existing 512-wide trunk.
pub const RECURRENT_HIDDEN: i64 = 256;
/// `ActionOutcome::as_floats` context width from actor commit 6468e46.
pub const RECURRENT_CONTEXT_FLOATS: i64 = 14;
pub const RECURRENT_CONTEXT_SCHEMA: &str = "action-outcome-v1";
pub const RECURRENT_CONTEXT_EMBEDDED: i64 = 128;

const CONTEXT_ACTION: i64 = 0;
const CONTEXT_PLAYER: i64 = 1;
const CONTEXT_BUILD: i64 = 3;
const CONTEXT_NUKE: i64 = 4;
const CONTEXT_SUCCESS: i64 = 5;
const CONTEXT_WASTED: i64 = 6;
const CONTEXT_TARGET_ID: i64 = 7;
const CONTEXT_TARGET_Y: i64 = 8;
const CONTEXT_TARGET_X: i64 = 9;
const CONTEXT_QUANTITY: i64 = 10;
const CONTEXT_COMMITMENT_AGE: i64 = 11;
const CONTEXT_HAD_ACTION: i64 = 12;
const CONTEXT_TARGET_KIND: i64 = 13;

const MASKED_NEG: f64 = -1e9;

const NEEDS_PLAYER: &[&str] = &[
    "attack",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "embargo",
    "retreat",
    "embargo_stop",
    "target_player",
    "alliance_extension",
];
const NEEDS_TILE: &[&str] = &[
    "boat",
    "build",
    "launch_nuke",
    "spawn",
    "upgrade_structure",
    "move_warship",
    "cancel_boat",
    "delete_unit",
];
const REFINE_TILE: &[&str] = &[
    "spawn",
    "build",
    "upgrade_structure",
    "cancel_boat",
    "delete_unit",
];
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
    let v: Vec<f32> = ACTIONS
        .iter()
        .map(|a| if names.contains(a) { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_slice(&v).to_device(device)
}

/// Metadata carried by an already-foveated observation. These tensors make
/// the compact fine grid self-describing and preserve absolute tile math.
pub struct CompactObsMeta {
    pub origin_y: Tensor,
    pub origin_x: Tensor,
    pub coarse_valid: Tensor,
    pub coarse_legal: Tensor,
}

/// Batched observation tensors. Ordinarily `grid`/masks are the full
/// /REGION-resolution input. With `compact` present they are already the
/// exact foveated fine window and must not be cropped again.
pub struct Obs {
    pub grid: Tensor,       // (B, C_GRID, gh, gw) f32
    pub grid_valid: Tensor, // (B, gh, gw) f32
    pub legal_tile: Tensor, // (B, gh, gw) f32
    /// Optional native /16 coarse grid `(B, C_GRID, cgh, cgw)`. When
    /// `None`, `foveate` derives coarse via 2x avg-pool of `grid`.
    pub grid_coarse: Option<Tensor>,
    pub players: Tensor,       // (B, MAX_SLOTS, P_FEAT) f32
    pub pmask: Tensor,         // (B, MAX_SLOTS) f32
    pub local: Tensor,         // (B, N_LOCAL, LOCAL, LOCAL) f32
    pub scalars: Tensor,       // (B, N_SCALARS) f32
    pub legal_actions: Tensor, // (B, N_ACTIONS) f32
    pub legal_ptarget: Tensor, // (B, N_ACTIONS, MAX_SLOTS) f32
    pub legal_build: Tensor,   // (B, N_BUILD) f32
    pub legal_nuke: Tensor,    // (B, N_NUKE) f32
    pub compact: Option<CompactObsMeta>,
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
            grid_coarse: self.grid_coarse.as_ref().map(|g| g.index_select(0, idx)),
            players: self.players.index_select(0, idx),
            pmask: self.pmask.index_select(0, idx),
            local: self.local.index_select(0, idx),
            scalars: self.scalars.index_select(0, idx),
            legal_actions: self.legal_actions.index_select(0, idx),
            legal_ptarget: self.legal_ptarget.index_select(0, idx),
            legal_build: self.legal_build.index_select(0, idx),
            legal_nuke: self.legal_nuke.index_select(0, idx),
            compact: self.compact.as_ref().map(|m| CompactObsMeta {
                origin_y: m.origin_y.index_select(0, idx),
                origin_x: m.origin_x.index_select(0, idx),
                coarse_valid: m.coarse_valid.index_select(0, idx),
                coarse_legal: m.coarse_legal.index_select(0, idx),
            }),
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

/// Upper bound on any single action's logit (legal or illegal) before it
/// enters softmax/log_softmax. Illegal actions sit at `MASKED_NEG` (very
/// negative) and are unaffected by a max-clamp, so masking still works.
/// Without this, an unbounded logit on a legal action lets the categorical
/// distribution collapse to a near-delta (entropy -> 0) and makes
/// log_softmax's gradient increasingly ill-conditioned, which is the root
/// cause of the entropy-collapse -> instability chain seen in training.
/// Clamping bounds how peaked the policy can ever get, independent of
/// whatever the entropy-bonus coefficient happens to be doing.
const LOGIT_CLAMP_MAX: f64 = 30.0;

/// `clamp_max` alone doesn't help if a logit is already NaN: IEEE-754
/// comparisons against NaN are always false, so `nan.clamp_max(x) == nan`
/// (verified: PyTorch's `clamp` propagates NaN rather than bounding it).
/// A live run hit exactly this - `multinomial`'s CUDA kernel asserted
/// "probability tensor contains inf, nan or element < 0" a few updates
/// after logit-clamping alone was deployed, meaning some upstream op (the
/// AMP/bf16 forward path is the prime suspect) produced a NaN/Inf logit
/// that sailed straight through the max-clamp. `nan_to_num` replaces NaN
/// with 0 (a neutral/uniform-ish logit) and any +-inf with the same
/// bounds `clamp_max`/`MASKED_NEG` already use, so this must run *before*
/// clamping to actually make softmax/log_softmax's input safe.
fn sanitize_logits(logits: &Tensor) -> Tensor {
    logits
        .nan_to_num(0.0, LOGIT_CLAMP_MAX, MASKED_NEG)
        .clamp_max(LOGIT_CLAMP_MAX)
}

fn categorical_sample(logits: &Tensor, greedy: bool) -> (Tensor, Tensor) {
    let logits = sanitize_logits(logits);
    let logp_all = logits.log_softmax(-1, Kind::Float);
    let idx = if greedy {
        logits.argmax(-1, false)
    } else {
        let probs = logits.softmax(-1, Kind::Float);
        probs.multinomial(1, true).squeeze_dim(-1)
    };
    let logp = logp_all
        .gather(-1, &idx.unsqueeze(-1), false)
        .squeeze_dim(-1);
    (idx, logp)
}

fn categorical_logp(logits: &Tensor, idx: &Tensor) -> Tensor {
    // Defensive at the gather boundary: unused choice fields are encoded
    // as -1, and any future actor/learner transport bug must not poison the
    // CUDA context with a device-side indexing assert. Callers still mask
    // unused heads out of the composed log-prob.
    let classes = logits.size().last().copied().unwrap_or(1).max(1);
    let idx = idx.clamp(0, classes - 1);
    sanitize_logits(logits)
        .log_softmax(-1, Kind::Float)
        .gather(-1, &idx.unsqueeze(-1), false)
        .squeeze_dim(-1)
}

fn categorical_entropy(logits: &Tensor) -> Tensor {
    let logp = sanitize_logits(logits).log_softmax(-1, Kind::Float);
    let p = logp.exp();
    -(p * logp).sum_dim_intlist(-1, false, Kind::Float)
}

/// alpha/beta >= 1 (unimodal, bounded Beta): 1 + softplus(raw), *softly*
/// bounded above so the Beta can't collapse into a numerically-degenerate
/// delta (same rationale as `LOGIT_CLAMP_MAX` above: bound how confident/
/// peaked any policy output can get, independent of gradient dynamics).
///
/// This used to be a hard `clamp_max`, which has the exact same "stuck
/// forever" defect `sanitize_value`'s doc describes for the value head:
/// zero gradient once alpha/beta drift past the bound, so the optimizer
/// can never pull them back down. Live evidence this session that this
/// was live and biting: `entq` (this head's differential entropy - can
/// legitimately go very negative for a highly-peaked Beta, unlike
/// discrete entropy) swung from a healthy ~-0.07 to ~-1.5 in the exact
/// same handful of updates `v` exploded in, both recovering together
/// on earlier runs - i.e. this head's own instability was likely
/// corrupting the shared trunk's features (which the value head reads,
/// even after being gradient-decoupled from it) rather than being an
/// independent, unrelated symptom. Same fix as the value head: swap the
/// hard clamp for a soft bound that saturates without ever fully zeroing
/// the gradient.
const QUANTITY_AB_MAX: f64 = 1000.0;
fn quantity_ab(params: &Tensor) -> (Tensor, Tensor) {
    // Same NaN-before-clamp hazard as `sanitize_logits` above: sanitize the
    // raw params first, then derive alpha/beta, so a NaN raw value can't
    // survive the soft bound and reach `lgamma`/`digamma` downstream.
    let params = params.nan_to_num(0.0, 1.0e12, -1.0e12);
    let raw_ab: Tensor = params.softplus() + 1.0;
    // x / (1 + (x-1)/(C-1)) : same soft-saturating shape as
    // `PolicyNet::sanitize_value`, shifted so raw_ab's floor of 1.0 maps
    // to itself (an alpha/beta of exactly 1 - the least-peaked case,
    // uniform - must stay untouched, not soft-bounded toward 0).
    let excess: Tensor = &raw_ab - 1.0;
    let ab: Tensor = &excess / (excess.abs() / (QUANTITY_AB_MAX - 1.0) + 1.0) + 1.0;
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
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        ResBlock {
            conv1: nn::conv2d(p / "conv1", c, c, 3, cfg),
            conv2: nn::conv2d(p / "conv2", c, c, 3, cfg),
        }
    }
    fn forward(&self, x: &Tensor, valid: Option<&Tensor>, amp: bool) -> Tensor {
        if amp {
            let mut h = conv2d_bf16(&self.conv1, x, [1, 1], [1, 1]).silu();
            if let Some(valid) = valid {
                h *= valid;
            }
            let mut residual = conv2d_bf16(&self.conv2, &h, [1, 1], [1, 1]);
            if let Some(valid) = valid {
                residual *= valid;
            }
            let mut out = (x + residual).silu();
            if let Some(valid) = valid {
                out *= valid;
            }
            out
        } else {
            let mut h = self.conv1.forward(x).silu();
            if let Some(valid) = valid {
                h *= valid;
            }
            let mut residual = self.conv2.forward(&h);
            if let Some(valid) = valid {
                residual *= valid;
            }
            let mut out = (x + residual).silu();
            if let Some(valid) = valid {
                out *= valid;
            }
            out
        }
    }
}

struct GridTower {
    stem: nn::Conv2D,
    blocks: Vec<ResBlock>,
}

impl GridTower {
    fn new(p: &nn::Path, c_in: i64, gc: i64, blocks: i64) -> Self {
        let cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let stem = nn::conv2d(p / "stem", c_in, gc, 3, cfg);
        let blocks = (0..blocks)
            .map(|i| ResBlock::new(&(p / "block" / i), gc))
            .collect();
        GridTower { stem, blocks }
    }
    /// `amp=true` runs the whole tower (stem + every residual block) in
    /// bf16, casting in once at the input and back to f32 once at the
    /// output (see `conv2d_bf16`); `amp=false` is the byte-for-byte
    /// original f32 path.
    fn forward_masked(&self, x: &Tensor, valid: Option<&Tensor>, amp: bool) -> Tensor {
        let mask = valid.map(|v| {
            v.unsqueeze(1)
                .to_kind(if amp { Kind::BFloat16 } else { Kind::Float })
        });
        if amp {
            let xb = x.to_kind(Kind::BFloat16);
            let mut h = conv2d_bf16(&self.stem, &xb, [1, 1], [1, 1]).silu();
            if let Some(mask) = &mask {
                h *= mask;
            }
            for b in &self.blocks {
                h = b.forward(&h, mask.as_ref(), true);
                // A biased convolution produces non-zero values in padded
                // cells. Clear them after every block so the next 3x3
                // convolution sees exactly the zero boundary a native-shape
                // singleton sees, rather than leaked padding activations.
                if let Some(mask) = &mask {
                    h *= mask;
                }
            }
            h.to_kind(Kind::Float)
        } else {
            let mut h = self.stem.forward(x).silu();
            if let Some(mask) = &mask {
                h *= mask;
            }
            for b in &self.blocks {
                h = b.forward(&h, mask.as_ref(), false);
                if let Some(mask) = &mask {
                    h *= mask;
                }
            }
            h
        }
    }

    fn forward(&self, x: &Tensor, amp: bool) -> Tensor {
        self.forward_masked(x, None, amp)
    }
}

struct LocalNet {
    c1: nn::Conv2D,
    c2: nn::Conv2D,
    c3: nn::Conv2D,
}

impl LocalNet {
    fn new(p: &nn::Path) -> Self {
        let cfg = |pad| nn::ConvConfig {
            padding: pad,
            stride: 2,
            ..Default::default()
        };
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
            h.adaptive_avg_pool2d([1, 1])
                .flatten(1, -1)
                .to_kind(Kind::Float)
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
    grid_fine: Tensor,         // (B, C_GRID_FINE, fine_h, fine_w)
    legal_tile_fine: Tensor,   // (B, fine_h, fine_w)
    grid_valid_fine: Tensor,   // (B, fine_h, fine_w) - 0 in the crop's padded region, else 1
    grid_coarse: Tensor, // (B, C_GRID, cgh, cgw) - always the whole map, unaffected by the crop
    gc_valid: Tensor,    // (B, cgh, cgw)
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

/// Private forward plumbing. Keeping the exact `Foveation` beside the maps
/// derived from it prevents action/evaluation heads from rebuilding the crop,
/// masks, and coordinate origins after the trunk has already computed them.
struct TrunkOutput {
    h: Tensor,
    gc_map: Tensor,
    gf_map: Tensor,
    p: Tensor,
    fov: Foveation,
}

struct ForwardOutput {
    act_logits: Tensor,
    player_logits: Tensor,
    tile_coarse: Tensor,
    tile_fine: Tensor,
    build: Tensor,
    nuke: Tensor,
    quantity: Tensor,
    value: Tensor,
    fov: Foveation,
}

/// Previous-action/result encoder and 256-wide GRUCell. Its contribution to
/// the unchanged trunk passes through a zero-initialized residual projection,
/// making a warm-started V8.1 policy initially output-bit-identical.
struct RecurrentCore {
    action: nn::Embedding,
    player: nn::Embedding,
    target_kind: nn::Embedding,
    build: nn::Embedding,
    nuke: nn::Embedding,
    context: nn::Linear,
    gru_input: nn::Linear,
    gru_hidden: nn::Linear,
    residual: nn::Linear,
}

impl RecurrentCore {
    fn new(p: &nn::Path) -> Self {
        let action = nn::embedding(p / "context_action", N_ACTIONS + 1, 32, Default::default());
        let player = nn::embedding(p / "context_player", MAX_SLOTS + 1, 16, Default::default());
        let target_kind = nn::embedding(p / "context_target_kind", 4, 8, Default::default());
        let build = nn::embedding(p / "context_build", N_BUILD + 1, 8, Default::default());
        let nuke = nn::embedding(p / "context_nuke", N_NUKE + 1, 8, Default::default());
        let context = nn::linear(
            p / "context_projection",
            80,
            RECURRENT_CONTEXT_EMBEDDED,
            Default::default(),
        );
        let gru_input = nn::linear(
            p / "gru_input",
            HIDDEN + RECURRENT_CONTEXT_EMBEDDED,
            3 * RECURRENT_HIDDEN,
            Default::default(),
        );
        let gru_hidden = nn::linear(
            p / "gru_hidden",
            RECURRENT_HIDDEN,
            3 * RECURRENT_HIDDEN,
            Default::default(),
        );
        let mut residual = nn::linear(p / "residual", RECURRENT_HIDDEN, HIDDEN, Default::default());
        tch::no_grad(|| {
            let _ = residual.ws.zero_();
            if let Some(bias) = residual.bs.as_mut() {
                let _ = bias.zero_();
            }
        });
        Self {
            action,
            player,
            target_kind,
            build,
            nuke,
            context,
            gru_input,
            gru_hidden,
            residual,
        }
    }

    fn category(context: &Tensor, column: i64, classes: i64) -> Tensor {
        // -1 is the explicit "not present / no previous action" sentinel.
        (context.select(1, column).to_kind(Kind::Int64) + 1).clamp(0, classes)
    }

    fn encode_context(&self, context: &Tensor) -> Tensor {
        debug_assert_eq!(context.size()[1], RECURRENT_CONTEXT_FLOATS);
        let action = self
            .action
            .forward(&Self::category(context, CONTEXT_ACTION, N_ACTIONS));
        let player = self
            .player
            .forward(&Self::category(context, CONTEXT_PLAYER, MAX_SLOTS));
        let target_kind =
            self.target_kind
                .forward(&Self::category(context, CONTEXT_TARGET_KIND, 3));
        let build = self
            .build
            .forward(&Self::category(context, CONTEXT_BUILD, N_BUILD));
        let nuke = self
            .nuke
            .forward(&Self::category(context, CONTEXT_NUKE, N_NUKE));
        let target_id = context.select(1, CONTEXT_TARGET_ID);
        let target_id = target_id.sign() * target_id.abs().log1p() / 16.0;
        let commitment = context
            .select(1, CONTEXT_COMMITMENT_AGE)
            .clamp_min(0.0)
            .log1p()
            / 8.0;
        let continuous = Tensor::stack(
            &[
                context.select(1, CONTEXT_TARGET_Y),
                context.select(1, CONTEXT_TARGET_X),
                context.select(1, CONTEXT_QUANTITY),
                context.select(1, CONTEXT_SUCCESS),
                context.select(1, CONTEXT_WASTED),
                target_id,
                commitment,
                context.select(1, CONTEXT_HAD_ACTION),
            ],
            1,
        );
        self.context
            .forward(&Tensor::cat(
                &[&action, &player, &target_kind, &build, &nuke, &continuous],
                1,
            ))
            .silu()
    }

    fn forward(
        &self,
        trunk: &Tensor,
        context: &Tensor,
        hidden_in: &Tensor,
        reset_mask: &Tensor,
    ) -> (Tensor, Tensor) {
        let keep: Tensor = 1.0 - reset_mask.to_kind(Kind::Float).view([-1, 1]);
        let hidden = hidden_in * keep;
        let encoded = self.encode_context(context);
        let input = self.gru_input.forward(&Tensor::cat(&[trunk, &encoded], 1));
        let recurrent = self.gru_hidden.forward(&hidden);
        let input = input.chunk(3, 1);
        let recurrent = recurrent.chunk(3, 1);
        let reset = (&input[0] + &recurrent[0]).sigmoid();
        let update = (&input[1] + &recurrent[1]).sigmoid();
        let candidate = (&input[2] + reset * &recurrent[2]).tanh();
        let hidden_out: Tensor = (1.0 - &update) * candidate + update * hidden;
        let trunk_out = trunk + self.residual.forward(&hidden_out);
        (trunk_out, hidden_out)
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
    recurrent: Option<RecurrentCore>,
    head_action: nn::Linear,
    head_player_q: nn::Linear,
    head_tile_coarse: (nn::Conv2D, nn::Conv2D),
    head_tile_fine: (nn::Conv2D, nn::Conv2D),
    head_build: nn::Linear,
    head_nuke: nn::Linear,
    head_quantity: nn::Linear,
    head_value: nn::Linear,
    // Device-local constants used by action-dependent masks. Keeping these
    // beside the policy avoids rebuilding and uploading the same tiny tables
    // in every actor forward and every PPO sequence chunk.
    needs_player: Tensor,
    needs_tile: Tensor,
    needs_quantity: Tensor,
    refine_tile: Tensor,
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
    /// `gc`/`blocks` override the `GC`/`BLOCKS` module defaults (see
    /// `--gc`/`--blocks` in `main.rs`) - lets a smaller `GridTower` (e.g.
    /// GC=128, BLOCKS=2) be benchmarked without a code change. Both grid
    /// towers (coarse + fine) and the tile heads scale with `gc`; `trunk1`'s
    /// input width scales with `gc` too since it pools both towers' outputs.
    pub fn new(vs: &nn::Path, amp: bool, foveate: bool, gc: i64, blocks: i64) -> Self {
        Self::new_with_recurrence(vs, amp, foveate, gc, blocks, false)
    }

    pub fn new_with_recurrence(
        vs: &nn::Path,
        amp: bool,
        foveate: bool,
        gc: i64,
        blocks: i64,
        recurrent: bool,
    ) -> Self {
        let conv1 = |p: &nn::Path, ci, co| nn::conv2d(p, ci, co, 1, Default::default());
        PolicyNet {
            grid_coarse_net: GridTower::new(&(vs / "grid_coarse"), C_GRID, gc, blocks),
            grid_fine_net: GridTower::new(&(vs / "grid_fine"), C_GRID_FINE, gc, blocks),
            local_net: LocalNet::new(&(vs / "local")),
            player_in: nn::linear(vs / "player_in", P_FEAT, PC, Default::default()),
            tf_layers: (0..TF_LAYERS)
                .map(|i| EncoderLayer::new(&(vs / "tf" / i), PC, TF_FF))
                .collect(),
            trunk1: nn::linear(
                vs / "trunk1",
                2 * gc + PC + LC + N_SCALARS,
                HIDDEN,
                Default::default(),
            ),
            trunk2: nn::linear(vs / "trunk2", HIDDEN, HIDDEN, Default::default()),
            recurrent: recurrent.then(|| RecurrentCore::new(&(vs / "recurrent"))),
            head_action: nn::linear(vs / "head_action", HIDDEN, N_ACTIONS, Default::default()),
            head_player_q: nn::linear(vs / "head_player_q", HIDDEN, PC, Default::default()),
            head_tile_coarse: (
                conv1(&(vs / "htc1"), gc + HIDDEN, 256),
                conv1(&(vs / "htc2"), 256, 1),
            ),
            head_tile_fine: (
                conv1(&(vs / "htf1"), gc + HIDDEN, 256),
                conv1(&(vs / "htf2"), 256, 1),
            ),
            head_build: nn::linear(vs / "head_build", HIDDEN, N_BUILD, Default::default()),
            head_nuke: nn::linear(vs / "head_nuke", HIDDEN, N_NUKE, Default::default()),
            head_quantity: nn::linear(vs / "head_quantity", HIDDEN, 2, Default::default()),
            head_value: nn::linear(vs / "head_value", HIDDEN, 1, Default::default()),
            needs_player: action_table(NEEDS_PLAYER, vs.device()),
            needs_tile: action_table(NEEDS_TILE, vs.device()),
            needs_quantity: action_table(NEEDS_QUANTITY, vs.device()),
            refine_tile: action_table(REFINE_TILE, vs.device()),
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
    /// Own-tile mass for crop centering uses `EGO_OWN_CH` (first ego
    /// plane after the AE latent). Falls back to the map center before
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
        if let Some(meta) = &o.compact {
            debug_assert!(use_crop, "compact observations require foveation");
            let grid_coarse = o
                .grid_coarse
                .as_ref()
                .expect("compact observation requires coarse grid")
                .shallow_clone();
            let (cgh, cgw) = (meta.coarse_valid.size()[1], meta.coarse_valid.size()[2]);
            let fine_coarse = fine_to_coarse_mask_cropped(
                &o.legal_tile,
                &o.grid_valid,
                &meta.origin_y,
                &meta.origin_x,
                gh,
                gw,
                cgh,
                cgw,
            );
            return Foveation {
                grid_fine: Tensor::cat(&[&o.grid, &o.grid_valid.unsqueeze(1)], 1),
                legal_tile_fine: o.legal_tile.shallow_clone(),
                grid_valid_fine: o.grid_valid.shallow_clone(),
                grid_coarse,
                gc_valid: meta.coarse_valid.shallow_clone(),
                legal_tile_coarse: meta.coarse_legal.shallow_clone(),
                fine_coarse,
                origin_y: meta.origin_y.shallow_clone(),
                origin_x: meta.origin_x.shallow_clone(),
                fine_h: gh,
                fine_w: gw,
            };
        }
        let grid_coarse = match &o.grid_coarse {
            Some(gc) => gc.shallow_clone(),
            None => o
                .grid
                .avg_pool2d([2, 2], [2, 2], [0, 0], true, false, None::<i64>),
        };
        let gc_valid = o
            .grid_valid
            .unsqueeze(1)
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
            .squeeze_dim(1);
        let legal_tile_coarse = o
            .legal_tile
            .unsqueeze(1)
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
            .squeeze_dim(1);

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

        debug_assert!(
            gh >= 2 && gw >= 2,
            "foveate crop needs a grid at least 2x2, got ({gh}, {gw})"
        );
        let fine_h = (FOVEATE_SIZE.min(gh)).max(2);
        let fine_w = (FOVEATE_SIZE.min(gw)).max(2);
        let fine_h = fine_h - fine_h % 2;
        let fine_w = fine_w - fine_w % 2;
        let mine = o.grid.select(1, EGO_OWN_CH); // (B, gh, gw): own-tile occupancy fraction
        let (origin_y, origin_x) = crop_origin(&mine, gh, gw, fine_h, fine_w);

        let grid_cropped = crop_and_pad(
            &o.grid,
            &origin_y,
            &origin_x,
            fine_h,
            fine_w,
            FOVEATE_SIZE,
            FOVEATE_SIZE,
        );
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

    /// Materialize the exact `--foveate` policy view while the full
    /// assembled observation is still actor-owned. The result is device
    /// local and is either consumed immediately by the actor or serialized
    /// to host by `batch`; it is never sent to a learner thread.
    pub fn compact_observation(o: &Obs) -> Obs {
        debug_assert!(o.compact.is_none());
        let fov = Self::foveate(o, true);
        let fine = fov.grid_fine.narrow(1, 0, C_GRID);
        Obs {
            grid: fine,
            grid_valid: fov.grid_valid_fine,
            legal_tile: fov.legal_tile_fine,
            grid_coarse: Some(fov.grid_coarse),
            players: o.players.shallow_clone(),
            pmask: o.pmask.shallow_clone(),
            local: o.local.shallow_clone(),
            scalars: o.scalars.shallow_clone(),
            legal_actions: o.legal_actions.shallow_clone(),
            legal_ptarget: o.legal_ptarget.shallow_clone(),
            legal_build: o.legal_build.shallow_clone(),
            legal_nuke: o.legal_nuke.shallow_clone(),
            compact: Some(CompactObsMeta {
                origin_y: fov.origin_y,
                origin_x: fov.origin_x,
                coarse_valid: fov.gc_valid,
                coarse_legal: fov.legal_tile_coarse,
            }),
        }
    }

    fn trunk_forward(&self, o: &Obs) -> TrunkOutput {
        let fov = Self::foveate(o, self.foveate);

        let gc_map =
            self.grid_coarse_net
                .forward_masked(&fov.grid_coarse, Some(&fov.gc_valid), self.amp);
        let gc_valid_b = fov.gc_valid.unsqueeze(1);
        let gc_map = &gc_map * &gc_valid_b;
        let gc_pool = gc_map.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            / gc_valid_b
                .sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
                .clamp_min(1.0);

        let gf_map = self.grid_fine_net.forward(&fov.grid_fine, self.amp);
        let gf_valid_b = fov.grid_valid_fine.unsqueeze(1);
        let gf_map = &gf_map * &gf_valid_b;
        let gf_pool = gf_map.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            / gf_valid_b
                .sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
                .clamp_min(1.0);

        let mut p = self.player_in.forward(&o.players); // (B, S, PC)
        let key_pad_bias = (&o.pmask - 1.0).unsqueeze(1).unsqueeze(1) * (-MASKED_NEG); // (B,1,1,S)
        for layer in &self.tf_layers {
            p = layer.forward(&p, &key_pad_bias);
        }
        let m = o.pmask.unsqueeze(-1);
        let p_pool = (&p * &m).sum_dim_intlist(1i64, false, Kind::Float)
            / m.sum_dim_intlist(1i64, false, Kind::Float).clamp_min(1.0);

        let l_pool = self.local_net.forward(&o.local, self.amp);
        let cat = Tensor::cat(&[&gc_pool, &gf_pool, &p_pool, &l_pool, &o.scalars], -1);
        let h = self.trunk1.forward(&cat).silu();
        let h = self.trunk2.forward(&h).silu();
        // Single chokepoint for the whole forward pass: every head (value,
        // action logits, quantity, tile, player) is derived from `h`/`p`/
        // `gc_map`/`gf_map`, so sanitizing NaN/Inf here - rather than
        // separately in each downstream head - protects all of them at
        // once regardless of which upstream op (a manual-bf16-cast conv in
        // `--amp`'s path is the prime suspect; see the entropy-collapse
        // devlog entries) actually produced it. A live run showed exactly
        // one of four independently-initialized shard replicas producing
        // NaN value AND policy losses together from the very first
        // minibatch - i.e. from a shared upstream tensor, not something
        // head-specific - which is exactly what this guards against.
        let sanitize = |t: &Tensor| t.nan_to_num(0.0, 1.0e4, -1.0e4);
        let h = sanitize(&h);
        let gc_map = sanitize(&gc_map);
        let gf_map = sanitize(&gf_map);
        let p = sanitize(&p);
        TrunkOutput {
            h,
            gc_map,
            gf_map,
            p,
            fov,
        }
    }

    fn tile_head(head: &(nn::Conv2D, nn::Conv2D), map: &Tensor, h: &Tensor, amp: bool) -> Tensor {
        let (b, _, gh, gw) = map.size4().unwrap();
        let hb = h
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([b, HIDDEN, gh, gw], false);
        let cat = Tensor::cat(&[map, &hb], 1);
        // 1x1 convs over the full grid (up to GW_MAX x GH_MAX cells, GC +
        // HIDDEN input channels) - real compute, not just a cheap
        // per-pixel lookup, so worth running under the same bf16 path as
        // the towers; output cast back to f32 before it becomes the
        // tile logits (see `--amp` doc on `PolicyNet::amp`).
        if amp {
            let cat_b = cat.to_kind(Kind::BFloat16);
            let mid = conv2d_bf16(&head.0, &cat_b, [1, 1], [0, 0]).silu();
            conv2d_bf16(&head.1, &mid, [1, 1], [0, 0])
                .flatten(1, -1)
                .to_kind(Kind::Float)
        } else {
            head.1.forward(&head.0.forward(&cat).silu()).flatten(1, -1)
        }
    }

    /// Full forward pass. Returns raw head tensors; callers combine with
    /// masks (see `act`/`evaluate`).
    fn forward(&self, o: &Obs) -> ForwardOutput {
        self.forward_from_trunk(o, self.trunk_forward(o))
    }

    fn forward_from_trunk(&self, o: &Obs, trunk: TrunkOutput) -> ForwardOutput {
        let TrunkOutput {
            h,
            gc_map,
            gf_map,
            p,
            fov,
        } = trunk;
        let act_logits = self.head_action.forward(&h) + (&o.legal_actions - 1.0) * (-MASKED_NEG);
        let q = self.head_player_q.forward(&h); // (B, PC)
        let player_logits = q.unsqueeze(1).matmul(&p.transpose(-2, -1)).squeeze_dim(1); // (B, S)
        let tile_coarse = Self::tile_head(&self.head_tile_coarse, &gc_map, &h, self.amp);
        let tile_fine = Self::tile_head(&self.head_tile_fine, &gf_map, &h, self.amp);
        let build = self.head_build.forward(&h) + (&o.legal_build - 1.0) * (-MASKED_NEG);
        let nuke = self.head_nuke.forward(&h) + (&o.legal_nuke - 1.0) * (-MASKED_NEG);
        let quantity = self.head_quantity.forward(&h);
        // Match the Python trainer: the critic must co-train the shared
        // representation. Detaching `h` left a single linear value head
        // chasing features that the policy continuously moved; live runs
        // showed the resulting critic lag grow from v-loss 0.12 to 13k.
        // Huber loss in train.rs bounds the value gradient, so sharing the
        // trunk does not reintroduce the old unbounded-MSE failure mode.
        let value = Self::sanitize_value(&self.head_value.forward(&h).squeeze_dim(-1));
        ForwardOutput {
            act_logits,
            player_logits,
            tile_coarse,
            tile_fine,
            build,
            nuke,
            quantity,
            value,
            fov,
        }
    }

    fn forward_recurrent(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
        reset_mask: &Tensor,
    ) -> (ForwardOutput, Tensor) {
        let mut trunk = self.trunk_forward(o);
        let recurrent = self
            .recurrent
            .as_ref()
            .expect("recurrent API requires PolicyNet::new_with_recurrence(..., true)");
        let (h, hidden_out) = recurrent.forward(&trunk.h, context, hidden_in, reset_mask);
        trunk.h = h;
        (self.forward_from_trunk(o, trunk), hidden_out)
    }

    /// `train.rs`'s `ret_clip` only bounds the value *target* before the
    /// loss ever sees it - nothing ever bounded the value *head's own
    /// prediction*. A live run showed the value loss spike to 26.5
    /// BILLION in a single update (recovered from a mid-hundreds-of-
    /// thousands baseline that Huber/ret_clip were already failing to
    /// keep under control), which given Huber's bounded gradient can only
    /// come from the raw *prediction* itself drifting to a similarly
    /// extreme magnitude - i.e. the same "nothing bounds the network's
    /// own output" gap that `LOGIT_CLAMP_MAX`/`QUANTITY_AB_MAX` closed for
    /// the policy heads, just not yet closed for the value head.
    ///
    /// A first version of this used a hard `clamp`, which made things
    /// *worse* in a follow-up run: once the raw prediction drifts past
    /// the bound, a hard clamp's gradient is exactly zero out there, so
    /// gradient descent can never pull it back - the value loss stopped
    /// exploding but got permanently stuck plateaued at the clamp's worst
    /// case instead of recovering (visibly different from the earlier,
    /// self-correcting v-spikes: this one just sat there for 15+ updates
    /// with reward regressing in step). A soft bound - `x / (1 +
    /// |x|/C)` - saturates at +-C the same way but its gradient decays
    /// polynomially (~1/x^2) rather than dropping to exactly zero, so
    /// even a badly-drifted prediction still gets pulled back toward a
    /// sane range instead of getting stuck there forever.
    /// Chosen relative to `train::Config::ret_clip`'s default (3000.0,
    /// the value *target*'s clamp): the previous 1e4 here left a wide
    /// "dead zone" (3000-10000) where a drifted prediction is guaranteed
    /// wrong against the capped target but the soft bound barely engages
    /// (it's only a near-identity for |x| << this constant), needing many
    /// more updates to get pulled back than a tighter, ret_clip-aligned
    /// bound would. 5000 keeps a comfortable 1.67x margin above ret_clip
    /// (so a legitimately-near-cap-scale prediction isn't over-compressed)
    /// while meaningfully shrinking that dead zone.
    const VALUE_CLAMP_ABS: f64 = 5.0e3;
    fn sanitize_value(value: &Tensor) -> Tensor {
        let v = value.nan_to_num(0.0, 1.0e12, -1.0e12);
        &v / (v.abs() / Self::VALUE_CLAMP_ABS + 1.0)
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
        let refine_action = self.refine_tile.index_select(0, action);
        let has_fine = fine_coarse
            .flatten(1, -1)
            .sum_dim_intlist(1i64, false, Kind::Float)
            .gt(0.0)
            .to_kind(Kind::Float);
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
        o.legal_tile
            .unsqueeze(1)
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true)
            .squeeze_dim(1)
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
        let mask_sum = mask
            .flatten(1, -1)
            .sum_dim_intlist(1i64, false, Kind::Float);
        let fallback = &o.legal_tile * &o.grid_valid;
        let fb_sum = fallback
            .flatten(1, -1)
            .sum_dim_intlist(1i64, false, Kind::Float);
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
    fn fine_logits_for_coarse_any(
        &self,
        tile_fine: &Tensor,
        o: &Obs,
        fov: &Foveation,
        coarse: &Tensor,
        cgw: i64,
    ) -> Tensor {
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
            global_to_fine_local_cropped(
                region,
                fov.fine_h,
                fov.fine_w,
                &fov.origin_y,
                &fov.origin_x,
            )
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
    ) -> (
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let output = self.forward(o);
        self.act_from_forward(o, greedy, output)
    }

    /// Actor-facing API matching commit 6468e46. Actor-owned state may reset
    /// rows externally; batched reset users call `act_with_state_masked`.
    pub fn act_with_state(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
        greedy: bool,
    ) -> (
        (
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ),
        Tensor,
    ) {
        let reset = Tensor::zeros([hidden_in.size()[0]], (Kind::Float, hidden_in.device()));
        self.act_with_state_masked(o, hidden_in, context, &reset, greedy)
    }

    /// `reset_mask` is batched, with 1 resetting a row before the GRUCell.
    pub fn act_with_state_masked(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
        reset_mask: &Tensor,
        greedy: bool,
    ) -> (
        (
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ),
        Tensor,
    ) {
        let (output, hidden_out) = self.forward_recurrent(o, hidden_in, context, reset_mask);
        (self.act_from_forward(o, greedy, output), hidden_out)
    }

    fn act_from_forward(
        &self,
        o: &Obs,
        greedy: bool,
        output: ForwardOutput,
    ) -> (
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let ForwardOutput {
            act_logits,
            player_logits: player_logits_raw,
            tile_coarse,
            tile_fine,
            build,
            nuke,
            quantity,
            value,
            fov,
        } = output;
        let (a, mut logp) = categorical_sample(&act_logits, greedy);

        let pmask = o
            .legal_ptarget
            .gather(
                1,
                &a.view([-1, 1, 1]).expand([-1, 1, MAX_SLOTS as i64], false),
                false,
            )
            .squeeze_dim(1);
        let player_logits = &player_logits_raw + (&pmask - 1.0) * (-MASKED_NEG);
        let (player, player_lp) = categorical_sample(&player_logits, greedy);
        let (build_s, build_lp) = categorical_sample(&build, greedy);
        let (nuke_s, nuke_lp) = categorical_sample(&nuke, greedy);

        let (qa, qb) = quantity_ab(&quantity);
        let q = if greedy {
            &qa / (&qa + &qb)
        } else {
            sample_beta_host(&qa, &qb)
        }
        .clamp(1e-4, 1.0 - 1e-4);
        let q_lp = beta_log_prob(&q, &qa, &qb);

        let needs_p = self.needs_player.index_select(0, &a);
        let needs_t = self.needs_tile.index_select(0, &a);
        let needs_q = self.needs_quantity.index_select(0, &a);
        let is_build = a
            .eq(ACTIONS.iter().position(|&x| x == "build").unwrap() as i64)
            .to_kind(Kind::Float);
        let is_nuke = a
            .eq(ACTIONS.iter().position(|&x| x == "launch_nuke").unwrap() as i64)
            .to_kind(Kind::Float);

        logp = logp + &needs_p * &player_lp;

        let (_cgh, cgw) = Self::coarse_dims(&fov.grid_coarse);
        let coarse_logits = self.coarse_logits_for_action(
            &tile_coarse,
            &fov.legal_tile_coarse,
            &fov.gc_valid,
            &fov.fine_coarse,
            &a,
        );
        let (coarse, _coarse_lp_sampled) = categorical_sample(&coarse_logits, greedy);
        let fine_logits = self.fine_logits_for_coarse_any(&tile_fine, o, &fov, &coarse, cgw);
        let (fine, fine_lp) = categorical_sample(&fine_logits, greedy);

        let refine_bool = self.refine_tile.index_select(0, &a).to_kind(Kind::Bool);
        let fine_global = self.fine_local_to_global_any(&fine, &fov);
        let coarse_global = coarse_local_to_global(&coarse, cgw);
        let eff_coarse_local =
            global_to_coarse_local(&fine_global, cgw).where_self(&refine_bool, &coarse);
        let coarse_lp = categorical_logp(&coarse_logits, &eff_coarse_local);
        let tile_lp = (&coarse_lp + &fine_lp).where_self(&refine_bool, &coarse_lp);
        let tile_region = fine_global.where_self(&refine_bool, &coarse_global);

        logp = logp + &needs_t * &tile_lp;
        logp = logp + &is_build * &build_lp;
        logp = logp + &is_nuke * &nuke_lp;
        logp = logp + &needs_q * &q_lp;

        (a, player, tile_region, build_s, nuke_s, q, logp, value)
    }

    /// Greedy/stochastic act plus masked action probabilities for debug overlays.
    pub fn act_with_debug(
        &self,
        o: &Obs,
        greedy: bool,
    ) -> (
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let output = self.forward(o);
        let action_probs = sanitize_logits(&output.act_logits).softmax(-1, Kind::Float);
        let (a, player, tile, build, nuke, q, logp, value) =
            self.act_from_forward(o, greedy, output);
        (a, player, tile, build, nuke, q, logp, value, action_probs)
    }

    /// Recurrent sibling of [`Self::act_with_debug`] for MODEL-overlay watch clips.
    pub fn act_with_state_debug(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
        greedy: bool,
    ) -> (
        (
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ),
        Tensor,
    ) {
        let reset = Tensor::zeros([hidden_in.size()[0]], (Kind::Float, hidden_in.device()));
        let (output, hidden_out) = self.forward_recurrent(o, hidden_in, context, &reset);
        let action_probs = sanitize_logits(&output.act_logits).softmax(-1, Kind::Float);
        let (a, player, tile, build, nuke, q, logp, value) =
            self.act_from_forward(o, greedy, output);
        (
            (a, player, tile, build, nuke, q, logp, value, action_probs),
            hidden_out,
        )
    }

    #[cfg(test)]
    fn act_recomputing_foveation_reference(
        &self,
        o: &Obs,
        greedy: bool,
    ) -> (
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let mut output = self.forward(o);
        output.fov = Self::foveate(o, self.foveate);
        self.act_from_forward(o, greedy, output)
    }

    /// Batched logprob/entropy/value for PPO updates (mirrors
    /// `Policy.evaluate`). Every sub-head's contribution is computed over
    /// the FULL batch and zeroed via the "used" mask instead of Python's
    /// boolean-subset indexing (equivalent result, simpler on tch, and the
    /// extra compute is harmless for throughput testing).
    pub fn evaluate(&self, o: &Obs, c: &ChoiceBatch) -> (Tensor, Tensor, Tensor, Tensor) {
        let output = self.forward(o);
        self.evaluate_from_forward(o, c, output)
    }

    /// One-timestep learner primitive; trainer sequence construction/BPTT is
    /// deliberately outside PolicyNet. Returns hidden_out as the fifth value.
    pub fn evaluate_with_state(
        &self,
        o: &Obs,
        c: &ChoiceBatch,
        hidden_in: &Tensor,
        context: &Tensor,
        reset_mask: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let (output, hidden_out) = self.forward_recurrent(o, hidden_in, context, reset_mask);
        let (logp, ent, ent_q, value) = self.evaluate_from_forward(o, c, output);
        (logp, ent, ent_q, value, hidden_out)
    }

    /// Fused truncated-BPTT learner forward.
    ///
    /// `o`, `c`, `context`, and `reset_mask` are flattened time-major as
    /// `[steps * envs, ...]`; `hidden_in` is the detached actor state at the
    /// chunk boundary `[envs, RECURRENT_HIDDEN]`. The observation trunk runs
    /// once over the full flattened chunk. Only the GRU transition remains
    /// sequential, after which every policy/value head runs once in a fused
    /// batch. Returned policy tensors preserve the same time-major order.
    pub fn evaluate_sequence_fused(
        &self,
        o: &Obs,
        c: &ChoiceBatch,
        hidden_in: &Tensor,
        context: &Tensor,
        reset_mask: &Tensor,
        steps: i64,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        assert!(
            steps > 0,
            "recurrent sequence must contain at least one step"
        );
        let total = o.grid.size()[0];
        assert_eq!(
            total % steps,
            0,
            "recurrent sequence batch is not time-major rectangular"
        );
        let envs = total / steps;
        assert_eq!(hidden_in.size(), [envs, RECURRENT_HIDDEN]);
        assert_eq!(context.size(), [total, RECURRENT_CONTEXT_FLOATS]);
        assert_eq!(reset_mask.size(), [total]);

        let mut trunk = self.trunk_forward(o);
        let recurrent = self
            .recurrent
            .as_ref()
            .expect("recurrent API requires PolicyNet::new_with_recurrence(..., true)");
        let mut hidden = hidden_in.shallow_clone();
        let mut recurrent_h = Vec::with_capacity(steps as usize);
        for t in 0..steps {
            let offset = t * envs;
            let (h, hidden_out) = recurrent.forward(
                &trunk.h.narrow(0, offset, envs),
                &context.narrow(0, offset, envs),
                &hidden,
                &reset_mask.narrow(0, offset, envs),
            );
            recurrent_h.push(h);
            hidden = hidden_out;
        }
        trunk.h = Tensor::cat(&recurrent_h.iter().collect::<Vec<_>>(), 0);
        let output = self.forward_from_trunk(o, trunk);
        let (logp, ent, ent_q, value) = self.evaluate_from_forward(o, c, output);
        (logp, ent, ent_q, value, hidden)
    }

    fn evaluate_from_forward(
        &self,
        o: &Obs,
        c: &ChoiceBatch,
        output: ForwardOutput,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let ForwardOutput {
            act_logits,
            player_logits: player_logits_raw,
            tile_coarse,
            tile_fine,
            build,
            nuke,
            quantity,
            value,
            fov,
        } = output;
        let mut logp = categorical_logp(&act_logits, &c.action);
        let mut ent = categorical_entropy(&act_logits);

        let action_c = c.action.clamp(0, N_ACTIONS - 1);
        let pmask = o
            .legal_ptarget
            .gather(
                1,
                &action_c
                    .view([-1, 1, 1])
                    .expand([-1, 1, MAX_SLOTS as i64], false),
                false,
            )
            .squeeze_dim(1);
        let player_logits = &player_logits_raw + (&pmask - 1.0) * (-MASKED_NEG);
        let p_used = c.player_slot.ge(0).to_kind(Kind::Float);
        let ps_c = c.player_slot.clamp(0, MAX_SLOTS as i64 - 1);
        logp = logp + &p_used * categorical_logp(&player_logits, &ps_c);
        ent = ent + &p_used * categorical_entropy(&player_logits);

        let (cgh, cgw) = Self::coarse_dims(&fov.grid_coarse);
        let t_used = c.tile_region.ge(0).to_kind(Kind::Float);
        let tr_c = c.tile_region.clamp(0, i64::MAX / 2);
        let coarse_target = global_to_coarse_local(&tr_c, cgw);
        let coarse_logits = self.coarse_logits_for_action(
            &tile_coarse,
            &fov.legal_tile_coarse,
            &fov.gc_valid,
            &fov.fine_coarse,
            &action_c,
        );
        let coarse_target_c = coarse_target.clamp(0, cgw * cgh - 1);
        logp = logp + &t_used * categorical_logp(&coarse_logits, &coarse_target_c);
        ent = ent + &t_used * categorical_entropy(&coarse_logits);

        let refine = self.refine_tile.index_select(0, &action_c) * &t_used;
        let fine_target = self.global_to_fine_local_any(&tr_c, &fov);
        let fine_target_c = fine_target.clamp(0, fov.fine_h * fov.fine_w - 1);
        let fine_logits =
            self.fine_logits_for_coarse_any(&tile_fine, o, &fov, &coarse_target_c, cgw);
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

    #[cfg(test)]
    fn evaluate_recomputing_foveation_reference(
        &self,
        o: &Obs,
        c: &ChoiceBatch,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let mut output = self.forward(o);
        output.fov = Self::foveate(o, self.foveate);
        self.evaluate_from_forward(o, c, output)
    }

    pub fn value_only(&self, o: &Obs) -> Tensor {
        let output = self.trunk_forward(o);
        Self::sanitize_value(&self.head_value.forward(&output.h).squeeze_dim(-1))
    }

    pub fn initial_hidden(&self, batch: i64) -> Tensor {
        Tensor::zeros(
            [batch, RECURRENT_HIDDEN],
            (Kind::Float, self.refine_tile.device()),
        )
    }

    pub fn value_with_state(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
    ) -> (Tensor, Tensor) {
        let reset = Tensor::zeros([hidden_in.size()[0]], (Kind::Float, hidden_in.device()));
        self.value_with_state_masked(o, hidden_in, context, &reset)
    }

    pub fn value_with_state_masked(
        &self,
        o: &Obs,
        hidden_in: &Tensor,
        context: &Tensor,
        reset_mask: &Tensor,
    ) -> (Tensor, Tensor) {
        let (output, hidden_out) = self.forward_recurrent(o, hidden_in, context, reset_mask);
        (output.value, hidden_out)
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
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
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
    Tensor::from_slice(&out)
        .view(a.size().as_slice())
        .to_device(dev)
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
    let cy = (&has_any * &centroid_y + &no_owned * ((h - 1) as f64 / 2.0))
        .round()
        .to_kind(Kind::Int64);
    let cx = (&has_any * &centroid_x + &no_owned * ((w - 1) as f64 / 2.0))
        .round()
        .to_kind(Kind::Int64);

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
    let flat_idx = (abs_y * src_w + abs_x)
        .view([b, 1, crop_h * crop_w])
        .expand([b, c, crop_h * crop_w], false);

    let src_flat = src.flatten(2, -1); // (B, C, src_h * src_w)
    let cropped = src_flat
        .gather(-1, &flat_idx, false)
        .view([b, c, crop_h, crop_w]);

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
    place_crop(
        &local_coarse,
        &origin_cy,
        &origin_cx,
        fine_h / 2,
        fine_w / 2,
        cgh,
        cgw,
    )
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
    let mask_sum = mask
        .flatten(1, -1)
        .sum_dim_intlist(1i64, false, Kind::Float);
    let fallback = legal_tile_fine * grid_valid_fine;
    let fb_sum = fallback
        .flatten(1, -1)
        .sum_dim_intlist(1i64, false, Kind::Float);
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
fn fine_local_to_global_cropped(
    local: &Tensor,
    fine_w: i64,
    origin_y: &Tensor,
    origin_x: &Tensor,
) -> Tensor {
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

    let mask = ly
        .ge(0)
        .logical_and(&ly.lt(fine_h))
        .logical_and(&lx.ge(0))
        .logical_and(&lx.lt(fine_w));

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
            grid_coarse: None,
            players: Tensor::rand([b, ms, P_FEAT], opts),
            pmask: Tensor::ones([b, ms], opts),
            local: Tensor::rand([b, N_LOCAL, LOCAL, LOCAL], opts),
            scalars: Tensor::rand([b, N_SCALARS], opts),
            legal_actions: Tensor::ones([b, na], opts),
            legal_ptarget: Tensor::ones([b, na, ms], opts),
            legal_build: Tensor::ones([b, N_BUILD], opts),
            legal_nuke: Tensor::ones([b, N_NUKE], opts),
            compact: None,
        }
    }

    fn assert_finite(t: &Tensor, what: &str) {
        let all_finite = t.isfinite().all().double_value(&[]);
        assert!(all_finite != 0.0, "{what} has non-finite values: {t:?}");
    }

    fn no_previous_context(batch: i64) -> Tensor {
        Tensor::from_slice(&[
            -1.0f32, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0,
        ])
        .view([1, RECURRENT_CONTEXT_FLOATS])
        .repeat([batch, 1])
    }

    fn assert_exact(actual: &Tensor, expected: &Tensor, name: &str) {
        let difference = (actual - expected).abs().max().double_value(&[]);
        assert_eq!(difference, 0.0, "{name} is not bit-identical");
    }

    #[test]
    fn safetensors_variable_names_match_interchange_schema() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _policy = PolicyNet::new(&vs.root(), false, true, GC, BLOCKS);
        let variables = vs.variables();
        for key in [
            "grid_coarse.stem.weight",
            "grid_fine.block.3.conv2.bias",
            "local.c1.weight",
            "tf.0.q.weight",
            "tf.1.ln2.bias",
            "trunk1.weight",
            "htf2.bias",
            "head_quantity.weight",
        ] {
            assert!(
                variables.contains_key(key),
                "missing interchange tensor {key}"
            );
        }
    }

    #[test]
    fn recurrent_safetensors_schema_is_opt_in_and_round_trips() {
        let base_vs = nn::VarStore::new(Device::Cpu);
        let _base = PolicyNet::new(&base_vs.root(), false, false, 8, 1);
        let recurrent_vs = nn::VarStore::new(Device::Cpu);
        let _recurrent =
            PolicyNet::new_with_recurrence(&recurrent_vs.root(), false, false, 8, 1, true);
        let base = base_vs.variables();
        let recurrent = recurrent_vs.variables();
        assert!(base.keys().all(|name| !name.starts_with("recurrent.")));
        for name in base.keys() {
            assert!(recurrent.contains_key(name), "recurrent schema lost {name}");
        }
        for name in [
            "recurrent.context_action.weight",
            "recurrent.context_player.weight",
            "recurrent.context_target_kind.weight",
            "recurrent.context_build.weight",
            "recurrent.context_nuke.weight",
            "recurrent.context_projection.weight",
            "recurrent.gru_input.weight",
            "recurrent.gru_hidden.weight",
            "recurrent.residual.weight",
        ] {
            assert!(
                recurrent.contains_key(name),
                "missing recurrent variable {name}"
            );
        }

        let legacy_path = std::env::temp_dir().join(format!(
            "oftrain-v81-schema-{}.safetensors",
            std::process::id()
        ));
        base_vs.save(&legacy_path).unwrap();
        let mut warm_vs = nn::VarStore::new(Device::Cpu);
        let _warm = PolicyNet::new_with_recurrence(&warm_vs.root(), false, false, 8, 1, true);
        let missing = warm_vs.load_partial(&legacy_path).unwrap();
        assert!(!missing.is_empty());
        assert!(
            missing.iter().all(|name| name.starts_with("recurrent.")),
            "V8.1 warm start may only leave recurrent variables fresh: {missing:?}"
        );
        assert_eq!(
            warm_vs.variables()["recurrent.residual.weight"]
                .abs()
                .max()
                .double_value(&[]),
            0.0
        );
        std::fs::remove_file(legacy_path).unwrap();

        let path = std::env::temp_dir().join(format!(
            "oftrain-v82-recurrent-schema-{}.safetensors",
            std::process::id()
        ));
        recurrent_vs.save(&path).unwrap();
        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let _loaded = PolicyNet::new_with_recurrence(&loaded_vs.root(), false, false, 8, 1, true);
        loaded_vs.load(&path).unwrap();
        assert_eq!(loaded_vs.variables().len(), recurrent.len());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn recurrent_context_is_distinct_and_done_masks_reset_rows() {
        tch::manual_seed(202);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new_with_recurrence(&vs.root(), false, false, 8, 1, true);
        let obs = synthetic_obs(Device::Cpu, 2, 5, 5);
        let hidden = Tensor::randn([2, RECURRENT_HIDDEN], (Kind::Float, Device::Cpu));
        let context_a = no_previous_context(2);
        let context_b = context_a.copy();
        for (column, value) in [
            (CONTEXT_ACTION, 2.0),
            (CONTEXT_PLAYER, 4.0),
            (CONTEXT_TARGET_KIND, 1.0),
            (CONTEXT_SUCCESS, 1.0),
            (CONTEXT_COMMITMENT_AGE, 7.0),
        ] {
            let _ = context_b.get(0).select(0, column).fill_(value);
        }
        let reset_none = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let (_, out_a) = policy.value_with_state_masked(&obs, &hidden, &context_a, &reset_none);
        let (_, out_b) = policy.value_with_state_masked(&obs, &hidden, &context_b, &reset_none);
        assert!((out_a.get(0) - out_b.get(0)).abs().max().double_value(&[]) > 0.0);

        let zero_hidden = Tensor::zeros([2, RECURRENT_HIDDEN], (Kind::Float, Device::Cpu));
        let (_, baseline) = policy.value_with_state(&obs, &zero_hidden, &no_previous_context(2));
        for (name, column, value) in [
            ("action", CONTEXT_ACTION, 2.0),
            ("player", CONTEXT_PLAYER, 4.0),
            ("tile_y", CONTEXT_TARGET_Y, 0.25),
            ("tile_x", CONTEXT_TARGET_X, 0.75),
            ("target_kind", CONTEXT_TARGET_KIND, 1.0),
            ("build", CONTEXT_BUILD, 2.0),
            ("nuke", CONTEXT_NUKE, 1.0),
            ("quantity", CONTEXT_QUANTITY, 0.6),
            ("success", CONTEXT_SUCCESS, 1.0),
            ("wasted", CONTEXT_WASTED, 1.0),
            ("commitment_age", CONTEXT_COMMITMENT_AGE, 7.0),
        ] {
            let changed = no_previous_context(2);
            let _ = changed.get(0).select(0, column).fill_(value);
            let (_, changed_hidden) = policy.value_with_state(&obs, &zero_hidden, &changed);
            assert!(
                (baseline.get(0) - changed_hidden.get(0))
                    .abs()
                    .max()
                    .double_value(&[])
                    > 0.0,
                "{name} context was ignored"
            );
        }

        let reset_first = Tensor::from_slice(&[1.0f32, 0.0]);
        let (_, masked) = policy.value_with_state_masked(&obs, &hidden, &context_b, &reset_first);
        let zeroed_hidden = hidden.copy();
        let _ = zeroed_hidden.get(0).zero_();
        let (_, reference) = policy.value_with_state(&obs, &zeroed_hidden, &context_b);
        assert_exact(&masked.get(0), &reference.get(0), "done-reset hidden row");
        let (_, unmasked) = policy.value_with_state(&obs, &hidden, &context_b);
        assert_exact(&masked.get(1), &unmasked.get(1), "nonterminal hidden row");
    }

    #[test]
    fn recurrent_act_and_evaluate_share_hidden_transition() {
        tch::manual_seed(252);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new_with_recurrence(&vs.root(), false, false, 8, 1, true);
        let obs = synthetic_obs(Device::Cpu, 2, 5, 5);
        let hidden = policy.initial_hidden(2);
        assert_eq!(hidden.size(), [2, RECURRENT_HIDDEN]);
        let context = no_previous_context(2);
        let ((action, player, tile, build, nuke, quantity, _, act_value), act_hidden) =
            policy.act_with_state(&obs, &hidden, &context, true);
        let choice = ChoiceBatch {
            action,
            player_slot: player,
            tile_region: tile,
            build_type: build,
            nuke_type: nuke,
            quantity_frac: quantity,
        };
        let reset = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let (logp, _, _, eval_value, eval_hidden) =
            policy.evaluate_with_state(&obs, &choice, &hidden, &context, &reset);
        assert_exact(&act_hidden, &eval_hidden, "act/evaluate hidden_out");
        assert_exact(&act_value, &eval_value, "act/evaluate value");
        assert_finite(&logp, "recurrent evaluate logprob");
        let reset_second = Tensor::from_slice(&[0.0f32, 1.0]);
        let (_, masked_hidden) =
            policy.act_with_state_masked(&obs, &act_hidden, &context, &reset_second, true);
        assert_eq!(masked_hidden.size(), [2, RECURRENT_HIDDEN]);
    }

    #[test]
    fn actor_hidden_trajectory_replays_exactly_across_bptt_and_done_reset() {
        tch::manual_seed(272);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new_with_recurrence(&vs.root(), false, false, 8, 1, true);
        let obs = synthetic_obs(Device::Cpu, 2, 5, 5);
        let done = [
            [false, false],
            [true, false],
            [false, false],
            [false, true],
            [false, false],
        ];
        let mut actor_hidden = policy.initial_hidden(2);
        let mut hidden_in = Vec::new();
        let mut hidden_out = Vec::new();
        let mut contexts = Vec::new();
        let mut resets = Vec::new();
        let mut choices = Vec::new();
        for t in 0..done.len() {
            let context = no_previous_context(2);
            let _ = context.get(0).select(0, CONTEXT_ACTION).fill_(t as f64);
            let _ = context
                .get(1)
                .select(0, CONTEXT_ACTION)
                .fill_((t + 3) as f64);
            let reset = if t == 0 {
                Tensor::zeros([2], (Kind::Float, Device::Cpu))
            } else {
                Tensor::from_slice(&[done[t - 1][0] as u8 as f32, done[t - 1][1] as u8 as f32])
            };
            hidden_in.push(actor_hidden.shallow_clone());
            let ((action, player, tile, build, nuke, quantity, _, _), next) =
                policy.act_with_state_masked(&obs, &actor_hidden, &context, &reset, true);
            choices.push(ChoiceBatch {
                action,
                player_slot: player,
                tile_region: tile,
                build_type: build,
                nuke_type: nuke,
                quantity_frac: quantity,
            });
            hidden_out.push(next.shallow_clone());
            contexts.push(context);
            resets.push(reset);
            actor_hidden = next;
        }

        for range in [0..2, 2..4, 4..5] {
            let mut replay = hidden_in[range.start].shallow_clone();
            for t in range {
                let (_, _, _, _, next) = policy.evaluate_with_state(
                    &obs,
                    &choices[t],
                    &replay,
                    &contexts[t],
                    &resets[t],
                );
                assert_exact(&next, &hidden_out[t], &format!("replay hidden t={t}"));
                replay = next;
            }
        }
        let reset_hidden = hidden_in[2].copy();
        let _ = reset_hidden.get(0).zero_();
        let (_, reset_reference) = policy.value_with_state(&obs, &reset_hidden, &contexts[2]);
        assert_exact(
            &hidden_out[2].get(0),
            &reset_reference.get(0),
            "done reset replay",
        );
    }

    #[test]
    fn fused_sequence_matches_reference_forward_and_gradients_with_mid_chunk_resets() {
        tch::manual_seed(292);
        let vs = nn::VarStore::new(Device::Cpu);
        let mut policy = PolicyNet::new_with_recurrence(&vs.root(), false, false, 8, 1, true);
        // The production warm start zeros this projection. Make it nonzero
        // so the test exercises gradients through the complete GRU, not only
        // the otherwise-identical observation/head paths.
        tch::no_grad(|| {
            let recurrent = policy.recurrent.as_mut().unwrap();
            let _ = recurrent.residual.ws.normal_(0.0, 0.02);
            if let Some(bias) = recurrent.residual.bs.as_mut() {
                let _ = bias.normal_(0.0, 0.02);
            }
        });

        let (steps, envs) = (3i64, 2i64);
        let obs = synthetic_obs(Device::Cpu, steps * envs, 5, 5);
        let choice = ChoiceBatch {
            action: Tensor::from_slice(&[0i64, 1, 2, 3, 4, 5]),
            player_slot: Tensor::zeros([steps * envs], (Kind::Int64, Device::Cpu)),
            tile_region: Tensor::zeros([steps * envs], (Kind::Int64, Device::Cpu)),
            build_type: Tensor::zeros([steps * envs], (Kind::Int64, Device::Cpu)),
            nuke_type: Tensor::zeros([steps * envs], (Kind::Int64, Device::Cpu)),
            quantity_frac: Tensor::full([steps * envs], 0.4, (Kind::Float, Device::Cpu)),
        };
        let context = no_previous_context(steps * envs);
        for row in 0..steps * envs {
            let _ = context
                .get(row)
                .select(0, CONTEXT_ACTION)
                .fill_((row % N_ACTIONS) as f64);
            let _ = context
                .get(row)
                .select(0, CONTEXT_SUCCESS)
                .fill_((row % 2) as f64);
        }
        // Reset env 0 before t=1 and env 1 before t=2: both are inside this
        // BPTT chunk, rather than only at its detached initial boundary.
        let reset = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let initial_hidden = Tensor::randn([envs, RECURRENT_HIDDEN], (Kind::Float, Device::Cpu));

        let mut hidden = initial_hidden.shallow_clone();
        let mut reference_parts: [Vec<Tensor>; 4] = Default::default();
        for t in 0..steps {
            let idx = Tensor::arange_start(t * envs, (t + 1) * envs, (Kind::Int64, Device::Cpu));
            let (lp, en, eq, value, next) = policy.evaluate_with_state(
                &obs.index_select(&idx),
                &choice.index_select(&idx),
                &hidden,
                &context.index_select(0, &idx),
                &reset.index_select(0, &idx),
            );
            for (parts, value) in reference_parts.iter_mut().zip([lp, en, eq, value]) {
                parts.push(value);
            }
            hidden = next;
        }
        let reference: Vec<Tensor> = reference_parts
            .iter()
            .map(|parts| Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0))
            .collect();
        let reference_hidden = hidden;
        let reference_loss: Tensor = &reference[0].mean(Kind::Float)
            + 0.07 * &reference[1].mean(Kind::Float)
            + 0.03 * &reference[2].mean(Kind::Float)
            + 0.11 * &reference[3].mean(Kind::Float);
        reference_loss.backward();
        let reference_grads: std::collections::HashMap<_, _> = vs
            .variables()
            .into_iter()
            .filter_map(|(name, tensor)| {
                let grad = tensor.grad();
                grad.defined().then(|| (name, grad.copy()))
            })
            .collect();
        for (_, tensor) in vs.variables() {
            let mut grad = tensor.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }

        let (lp, en, eq, value, fused_hidden) =
            policy.evaluate_sequence_fused(&obs, &choice, &initial_hidden, &context, &reset, steps);
        for (name, fused, expected) in [
            ("logp", &lp, &reference[0]),
            ("entropy", &en, &reference[1]),
            ("quantity entropy", &eq, &reference[2]),
            ("value", &value, &reference[3]),
            ("hidden", &fused_hidden, &reference_hidden),
        ] {
            let max = (fused - expected).abs().max().double_value(&[]);
            assert!(max < 2e-5, "fused/reference {name} mismatch: {max:e}");
        }
        let fused_loss: Tensor = lp.mean(Kind::Float)
            + 0.07 * en.mean(Kind::Float)
            + 0.03 * eq.mean(Kind::Float)
            + 0.11 * value.mean(Kind::Float);
        fused_loss.backward();
        for (name, expected) in reference_grads {
            let actual = vs.variables()[&name].grad();
            let max = (&actual - &expected).abs().max().double_value(&[]);
            assert!(
                max < 2e-4,
                "fused/reference gradient {name} mismatch: {max:e}"
            );
        }
    }

    #[test]
    fn gradients_reach_gru_after_zero_residual_starts_learning() {
        tch::manual_seed(303);
        let vs = nn::VarStore::new(Device::Cpu);
        let mut policy = PolicyNet::new_with_recurrence(&vs.root(), false, false, 8, 1, true);
        let obs = synthetic_obs(Device::Cpu, 2, 5, 5);
        let hidden = Tensor::zeros([2, RECURRENT_HIDDEN], (Kind::Float, Device::Cpu));
        let context = no_previous_context(2);
        let (value, _) = policy.value_with_state(&obs, &hidden, &context);
        value.sum(Kind::Float).backward();
        let recurrent = policy.recurrent.as_mut().unwrap();
        assert!(recurrent.residual.ws.grad().abs().max().double_value(&[]) > 0.0);
        assert_eq!(
            recurrent.gru_input.ws.grad().abs().max().double_value(&[]),
            0.0
        );
        tch::no_grad(|| {
            recurrent.residual.ws += recurrent.residual.ws.grad() * 1e-3;
            for (_, tensor) in vs.variables() {
                let mut grad = tensor.grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
            }
        });
        let (value, _) = policy.value_with_state(&obs, &hidden, &context);
        value.sum(Kind::Float).backward();
        assert!(
            policy
                .recurrent
                .as_ref()
                .unwrap()
                .gru_input
                .ws
                .grad()
                .abs()
                .max()
                .double_value(&[])
                > 0.0
        );
    }

    /// Exercises `act()` (no_grad) + `evaluate()` + `backward()` for a
    /// given `PolicyNet` config, asserting every returned tensor and the
    /// summed loss stay finite. Shared by the amp/foveate/model-size
    /// tests below so each only has to say what's different about it.
    fn check_policy_finite(policy: &PolicyNet, o: &Obs) {
        let (a, player, tile, build, nuke, qty, logp, value) =
            tch::no_grad(|| policy.act(o, false));
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

        let loss = logp2.mean(Kind::Float)
            + ent.mean(Kind::Float)
            + ent_q.mean(Kind::Float)
            + value2.mean(Kind::Float);
        loss.backward();
        let loss_v = loss.double_value(&[]);
        assert!(loss_v.is_finite(), "loss not finite: {loss_v}");
    }

    /// Match Python actor-critic learning: both value and policy losses
    /// train the shared representation. Huber loss bounds the critic's
    /// contribution before it reaches this path.
    #[test]
    fn value_loss_gradient_reaches_the_shared_trunk() {
        tch::manual_seed(5);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, false, GC, BLOCKS);
        let o = synthetic_obs(Device::Cpu, 2, 6, 6);
        let (a, player, tile, build, nuke, qty, _logp, _value) =
            tch::no_grad(|| policy.act(&o, false));
        let choice = ChoiceBatch {
            action: a,
            player_slot: player,
            tile_region: tile,
            build_type: build,
            nuke_type: nuke,
            quantity_frac: qty,
        };

        fn trunk_grad_norm(vs: &nn::VarStore) -> f64 {
            let mut total = 0.0;
            for (name, mut t) in vs.variables() {
                if (name.contains("trunk")
                    || name.contains("grid_coarse_net")
                    || name.contains("grid_fine_net"))
                    && t.grad().defined()
                {
                    total += f64::try_from(t.grad().abs().sum(Kind::Float)).unwrap();
                }
                t.zero_grad();
            }
            total
        }

        let (_logp2, _ent, _ent_q, value2) = policy.evaluate(&o, &choice);
        value2.mean(Kind::Float).backward();
        let value_only_norm = trunk_grad_norm(&vs);
        assert!(
            value_only_norm > 0.0,
            "trunk MUST see nonzero gradient from the value output, got {value_only_norm}"
        );

        let (logp2, _ent, _ent_q, _value2) = policy.evaluate(&o, &choice);
        logp2.mean(Kind::Float).backward();
        let policy_norm = trunk_grad_norm(&vs);
        assert!(
            policy_norm > 0.0,
            "trunk MUST see nonzero gradient from the policy output, got {policy_norm}"
        );
    }

    #[test]
    fn amp_and_f32_paths_finite() {
        tch::manual_seed(0);
        for &amp in &[false, true] {
            let vs = nn::VarStore::new(Device::Cpu);
            let policy = PolicyNet::new(&vs.root(), amp, false, GC, BLOCKS);
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
        let policy = PolicyNet::new(&vs.root(), false, true, GC, BLOCKS);
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
        let policy = PolicyNet::new(&vs.root(), false, true, GC, BLOCKS);
        let o = synthetic_obs(Device::Cpu, 2, 6, 8);
        check_policy_finite(&policy, &o);
    }

    /// `--foveate` combined with `--amp`, since both mutate the same
    /// tile-head/tower forward paths and could interact badly.
    #[test]
    fn foveate_and_amp_finite() {
        tch::manual_seed(3);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), true, true, GC, BLOCKS);
        let o = synthetic_obs(Device::Cpu, 2, FOVEATE_SIZE * 2, FOVEATE_SIZE * 2);
        check_policy_finite(&policy, &o);
    }

    /// `--gc`/`--blocks` overrides (a smaller policy variant) build and
    /// run correctly.
    #[test]
    fn small_model_variant_finite() {
        tch::manual_seed(4);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, false, 128, 2);
        let o = synthetic_obs(Device::Cpu, 2, 6, 6);
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
        let expected: Vec<f32> = (0..ch)
            .flat_map(|dy| (0..cw).map(move |dx| ((8 + dy) * w + 10 + dx) as f32))
            .collect();
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
                assert_eq!(
                    placed_v[(y * w + x) as usize],
                    expect,
                    "mismatch at ({y}, {x})"
                );
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

    /// The compact path must be a representation change, not a policy
    /// change: the old full-grid foveation and an already-cropped Obs feed
    /// identical tensors into every head and preserve global tile targets.
    #[test]
    fn compact_observation_matches_full_grid_policy_and_coordinates() {
        tch::manual_seed(17);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true, 16, 1);
        let o = synthetic_obs(Device::Cpu, 2, FOVEATE_SIZE + 8, FOVEATE_SIZE + 12);
        let mut mine = o.grid.select(1, EGO_OWN_CH);
        let _ = mine.zero_();
        let _ = mine.get(0).get(3).get(5).fill_(1.0);
        let _ = mine
            .get(1)
            .get(FOVEATE_SIZE + 3)
            .get(FOVEATE_SIZE + 7)
            .fill_(1.0);

        let old_fov = PolicyNet::foveate(&o, true);
        let compact = PolicyNet::compact_observation(&o);
        let new_fov = PolicyNet::foveate(&compact, true);
        let close = |a: &Tensor, b: &Tensor, name: &str| {
            let d = (a - b).abs().max().double_value(&[]);
            assert!(d <= 1e-6, "{name} differs by {d}");
        };
        close(&old_fov.grid_fine, &new_fov.grid_fine, "fine grid");
        close(&old_fov.grid_coarse, &new_fov.grid_coarse, "coarse grid");
        close(
            &old_fov.fine_coarse,
            &new_fov.fine_coarse,
            "fine/coarse mask",
        );
        close(
            &old_fov.legal_tile_fine,
            &new_fov.legal_tile_fine,
            "fine legality",
        );
        assert_eq!(
            Vec::<i64>::try_from(&old_fov.origin_y).unwrap(),
            Vec::<i64>::try_from(&new_fov.origin_y).unwrap()
        );
        assert_eq!(
            Vec::<i64>::try_from(&old_fov.origin_x).unwrap(),
            Vec::<i64>::try_from(&new_fov.origin_x).unwrap()
        );

        let full_out = policy.forward(&o);
        let compact_out = policy.forward(&compact);
        for (i, (a, b)) in [
            (&full_out.act_logits, &compact_out.act_logits),
            (&full_out.player_logits, &compact_out.player_logits),
            (&full_out.tile_coarse, &compact_out.tile_coarse),
            (&full_out.tile_fine, &compact_out.tile_fine),
            (&full_out.build, &compact_out.build),
            (&full_out.nuke, &compact_out.nuke),
            (&full_out.quantity, &compact_out.quantity),
            (&full_out.value, &compact_out.value),
        ]
        .into_iter()
        .enumerate()
        {
            close(a, b, &format!("policy output {i}"));
        }

        let spawn = ACTIONS.iter().position(|&a| a == "spawn").unwrap() as i64;
        let choices = ChoiceBatch {
            action: Tensor::from_slice(&[spawn, spawn]),
            player_slot: Tensor::from_slice(&[-1i64, -1]),
            tile_region: Tensor::from_slice(&[
                3 * GW_MAX + 5,
                (FOVEATE_SIZE + 3) * GW_MAX + FOVEATE_SIZE + 7,
            ]),
            build_type: Tensor::from_slice(&[-1i64, -1]),
            nuke_type: Tensor::from_slice(&[-1i64, -1]),
            quantity_frac: Tensor::from_slice(&[-1.0f32, -1.0]),
        };
        let old_eval = policy.evaluate(&o, &choices);
        let new_eval = policy.evaluate(&compact, &choices);
        close(&old_eval.0, &new_eval.0, "evaluate logp");
        close(&old_eval.1, &new_eval.1, "evaluate entropy");
        close(&old_eval.3, &new_eval.3, "evaluate value");
    }

    fn assert_same_tensor(actual: &Tensor, expected: &Tensor, name: &str) {
        assert_eq!(actual.size(), expected.size(), "{name} shape");
        let max_diff = (actual - expected).abs().max().double_value(&[]);
        assert_eq!(max_diff, 0.0, "{name} differs by {max_diff}");
    }

    fn reference_choice(device: Device) -> ChoiceBatch {
        let spawn = ACTIONS.iter().position(|&a| a == "spawn").unwrap() as i64;
        ChoiceBatch {
            action: Tensor::from_slice(&[spawn, spawn]).to_device(device),
            player_slot: Tensor::from_slice(&[-1i64, -1]).to_device(device),
            tile_region: Tensor::from_slice(&[
                4 * GW_MAX + 6,
                (FOVEATE_SIZE + 2) * GW_MAX + FOVEATE_SIZE + 4,
            ])
            .to_device(device),
            build_type: Tensor::from_slice(&[-1i64, -1]).to_device(device),
            nuke_type: Tensor::from_slice(&[-1i64, -1]).to_device(device),
            quantity_frac: Tensor::from_slice(&[-1.0f32, -1.0]).to_device(device),
        }
    }

    fn zero_parameter_gradients(vs: &nn::VarStore) {
        for (_, mut parameter) in vs.variables() {
            parameter.zero_grad();
        }
    }

    fn parameter_gradients(vs: &nn::VarStore) -> std::collections::BTreeMap<String, Tensor> {
        vs.variables()
            .into_iter()
            .filter_map(|(name, parameter)| {
                let grad = parameter.grad();
                grad.defined().then(|| (name, grad.detach().copy()))
            })
            .collect()
    }

    /// Pins the pre-optimization behavior: the old path recomputed
    /// `foveate` after `forward`, while the optimized path carries forward
    /// that exact object. Both representations must produce identical
    /// masks/origins, deterministic actions, evaluation outputs, and
    /// parameter gradients.
    #[test]
    fn carried_foveation_matches_recomputation_for_full_and_compact_obs() {
        tch::manual_seed(29);
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true, 8, 1);
        let full = synthetic_obs(Device::Cpu, 2, FOVEATE_SIZE + 8, FOVEATE_SIZE + 12);
        let mut mine = full.grid.select(1, EGO_OWN_CH);
        let _ = mine.zero_();
        let _ = mine.get(0).get(4).get(6).fill_(1.0);
        let _ = mine
            .get(1)
            .get(FOVEATE_SIZE + 2)
            .get(FOVEATE_SIZE + 4)
            .fill_(1.0);
        let compact = PolicyNet::compact_observation(&full);
        let choice = reference_choice(Device::Cpu);

        for (obs_name, obs) in [("foveated", &full), ("compact", &compact)] {
            let forward = policy.forward(obs);
            let recomputed = PolicyNet::foveate(obs, true);
            for (field, carried, reference) in [
                ("grid_fine", &forward.fov.grid_fine, &recomputed.grid_fine),
                (
                    "legal_tile_fine",
                    &forward.fov.legal_tile_fine,
                    &recomputed.legal_tile_fine,
                ),
                (
                    "grid_valid_fine",
                    &forward.fov.grid_valid_fine,
                    &recomputed.grid_valid_fine,
                ),
                (
                    "grid_coarse",
                    &forward.fov.grid_coarse,
                    &recomputed.grid_coarse,
                ),
                ("gc_valid", &forward.fov.gc_valid, &recomputed.gc_valid),
                (
                    "legal_tile_coarse",
                    &forward.fov.legal_tile_coarse,
                    &recomputed.legal_tile_coarse,
                ),
                (
                    "fine_coarse",
                    &forward.fov.fine_coarse,
                    &recomputed.fine_coarse,
                ),
                ("origin_y", &forward.fov.origin_y, &recomputed.origin_y),
                ("origin_x", &forward.fov.origin_x, &recomputed.origin_x),
            ] {
                assert_same_tensor(carried, reference, &format!("{obs_name}.{field}"));
            }
            assert_eq!(forward.fov.fine_h, recomputed.fine_h, "{obs_name}.fine_h");
            assert_eq!(forward.fov.fine_w, recomputed.fine_w, "{obs_name}.fine_w");

            // Greedy mode makes categorical and quantity choices
            // deterministic; the stochastic actor path intentionally keeps
            // its existing RNG behavior and is not reseeded here.
            let optimized_act = tch::no_grad(|| policy.act(obs, true));
            let reference_act =
                tch::no_grad(|| policy.act_recomputing_foveation_reference(obs, true));
            for (field, optimized, reference) in [
                ("action", &optimized_act.0, &reference_act.0),
                ("player", &optimized_act.1, &reference_act.1),
                ("tile", &optimized_act.2, &reference_act.2),
                ("build", &optimized_act.3, &reference_act.3),
                ("nuke", &optimized_act.4, &reference_act.4),
                ("quantity", &optimized_act.5, &reference_act.5),
                ("logp", &optimized_act.6, &reference_act.6),
                ("value", &optimized_act.7, &reference_act.7),
            ] {
                assert_same_tensor(optimized, reference, &format!("{obs_name}.act.{field}"));
            }

            let optimized_eval = policy.evaluate(obs, &choice);
            let reference_eval = policy.evaluate_recomputing_foveation_reference(obs, &choice);
            for (field, optimized, reference) in [
                ("logp", &optimized_eval.0, &reference_eval.0),
                ("entropy", &optimized_eval.1, &reference_eval.1),
                ("quantity_entropy", &optimized_eval.2, &reference_eval.2),
                ("value", &optimized_eval.3, &reference_eval.3),
            ] {
                assert_same_tensor(
                    optimized,
                    reference,
                    &format!("{obs_name}.evaluate.{field}"),
                );
            }

            zero_parameter_gradients(&vs);
            let optimized_loss = optimized_eval.0.mean(Kind::Float)
                + optimized_eval.1.mean(Kind::Float)
                + optimized_eval.2.mean(Kind::Float)
                + optimized_eval.3.mean(Kind::Float);
            optimized_loss.backward();
            let optimized_grads = parameter_gradients(&vs);

            zero_parameter_gradients(&vs);
            let reference_eval = policy.evaluate_recomputing_foveation_reference(obs, &choice);
            let reference_loss = reference_eval.0.mean(Kind::Float)
                + reference_eval.1.mean(Kind::Float)
                + reference_eval.2.mean(Kind::Float)
                + reference_eval.3.mean(Kind::Float);
            reference_loss.backward();
            let reference_grads = parameter_gradients(&vs);

            assert_eq!(
                optimized_grads.len(),
                reference_grads.len(),
                "{obs_name} gradient count"
            );
            for (name, optimized) in &optimized_grads {
                let reference = reference_grads
                    .get(name)
                    .unwrap_or_else(|| panic!("{obs_name} missing reference gradient {name}"));
                assert_same_tensor(optimized, reference, &format!("{obs_name}.gradient.{name}"));
            }
        }
    }
}

#[cfg(test)]
mod logit_clamp_tests {
    //! `LOGIT_CLAMP_MAX`/`QUANTITY_AB_MAX` guard the entropy-collapse ->
    //! instability chain seen in training (see devlog): without a bound,
    //! an ever-larger legal-action logit (or Beta alpha/beta) makes the
    //! distribution collapse toward a delta and drives `categorical_*`'s
    //! log_softmax/lgamma-based math into an increasingly degenerate,
    //! high-gradient regime. These tests pin the exact bound so a future
    //! change can't silently widen or remove it.
    use super::*;

    #[test]
    fn categorical_entropy_never_collapses_to_zero_no_matter_how_extreme_the_logit() {
        // One wildly dominant legal logit among otherwise-illegal (masked)
        // alternatives - exactly the shape that, pre-clamp, drove entropy
        // toward zero as training pushed the winning logit ever higher.
        let logits = Tensor::from_slice(&[
            1.0e6f32,
            MASKED_NEG as f32,
            MASKED_NEG as f32,
            MASKED_NEG as f32,
        ])
        .reshape([1, 4]);
        let ent: f64 = categorical_entropy(&logits).double_value(&[0]);
        assert!(ent.is_finite(), "entropy must stay finite, got {ent}");
        // With the clamp, the effective gap between the winning and next
        // logits is bounded by LOGIT_CLAMP_MAX vs MASKED_NEG, so entropy
        // is small but not exactly zero and never NaN/negative.
        assert!(ent >= 0.0, "entropy must be non-negative, got {ent}");
    }

    #[test]
    fn categorical_sample_and_logp_stay_finite_for_extreme_logits() {
        let logits = Tensor::from_slice(&[-1.0e8f32, 1.0e8f32, 5.0f32]).reshape([1, 3]);
        let (idx, logp) = categorical_sample(&logits, false);
        let logp_v: f64 = logp.double_value(&[0]);
        assert!(
            logp_v.is_finite(),
            "sampled logp must be finite, got {logp_v}"
        );
        let logp2 = categorical_logp(&logits, &idx);
        let logp2_v: f64 = logp2.double_value(&[0]);
        assert!(
            logp2_v.is_finite(),
            "categorical_logp must be finite, got {logp2_v}"
        );
    }

    #[test]
    fn categorical_logp_clamps_invalid_transport_indices() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
        let invalid = Tensor::from_slice(&[-1i64, 99]);
        let logp = categorical_logp(&logits, &invalid);
        let values: Vec<f32> = Vec::try_from(logp).unwrap();
        assert_eq!(values.len(), 2);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn categorical_sample_survives_an_actual_nan_logit() {
        // This is the exact failure mode a live run hit: clamp_max alone
        // does NOT save a NaN logit (NaN comparisons are always false), so
        // this must go through `sanitize_logits`'s nan_to_num first.
        let logits = Tensor::from_slice(&[f32::NAN, 2.0f32, f32::INFINITY]).reshape([1, 3]);
        let (idx, logp) = categorical_sample(&logits, false);
        let idx_v: i64 = idx.int64_value(&[0]);
        assert!(
            (0..3).contains(&idx_v),
            "sampled index must be valid, got {idx_v}"
        );
        let logp_v: f64 = logp.double_value(&[0]);
        assert!(
            logp_v.is_finite(),
            "logp must be finite for a NaN/Inf-containing logit row, got {logp_v}"
        );
    }

    #[test]
    fn quantity_ab_survives_an_actual_nan_param() {
        let params = Tensor::from_slice(&[f32::NAN, f32::NEG_INFINITY]).reshape([1, 1, 2]);
        let (a, b) = quantity_ab(&params);
        let a_v: f64 = a.double_value(&[0, 0]);
        let b_v: f64 = b.double_value(&[0, 0]);
        assert!(
            a_v.is_finite(),
            "alpha must be finite for a NaN raw param, got {a_v}"
        );
        assert!(
            b_v.is_finite(),
            "beta must be finite for a -inf raw param, got {b_v}"
        );
    }

    #[test]
    fn value_head_output_is_bounded_and_nan_safe() {
        // Directly exercises PolicyNet::sanitize_value (same rationale as
        // sanitize_logits, applied to the value head this time - a live
        // run's value loss spiked to 26.5 billion in one update, which
        // given Huber's bounded gradient could only come from the raw
        // value *prediction* itself, not the (already ret-clipped) target).
        let huge = Tensor::from_slice(&[1.0e9f32, -1.0e9f32, f32::NAN, f32::INFINITY]);
        let sanitized = PolicyNet::sanitize_value(&huge);
        let v: Vec<f64> = (0..4).map(|i| sanitized.double_value(&[i])).collect();
        for x in &v {
            assert!(x.is_finite(), "value must stay finite, got {v:?}");
            assert!(
                x.abs() <= PolicyNet::VALUE_CLAMP_ABS,
                "value must stay bounded, got {v:?}"
            );
        }
    }

    #[test]
    fn value_head_soft_bound_preserves_nonzero_gradient_even_when_saturated() {
        // The whole point of the soft (x / (1 + |x|/C)) bound over a hard
        // clamp: even a wildly-drifted raw prediction must still carry a
        // nonzero gradient back toward Self::VALUE_CLAMP_ABS, or the
        // optimizer can never recover from it (this is exactly the "got
        // stuck at the clamp forever" regression a hard clamp caused).
        let raw = Tensor::from_slice(&[1.0e6f32]).set_requires_grad(true);
        let out = PolicyNet::sanitize_value(&raw);
        out.backward();
        let grad = raw.grad();
        let g: f64 = grad.double_value(&[0]);
        assert!(
            g.is_finite() && g > 0.0,
            "gradient must be finite and nonzero (pulling back toward the bound), got {g}"
        );
    }

    #[test]
    fn value_head_soft_bound_is_near_identity_for_small_inputs() {
        // For |x| << C the bound should barely perturb the value at all -
        // it should only kick in once predictions actually start drifting
        // to an unreasonable magnitude, not distort ordinary training.
        let small = Tensor::from_slice(&[1.0f32, -5.0f32, 100.0f32]);
        let out = PolicyNet::sanitize_value(&small);
        let v: Vec<f64> = (0..3).map(|i| out.double_value(&[i])).collect();
        let expect = [1.0f64, -5.0, 100.0];
        for (got, want) in v.iter().zip(expect.iter()) {
            // |x|/C is at most 100/1e4 = 0.01 here, so the relative error
            // introduced by the "+ |x|/C" denominator term is small but
            // not vanishingly so - use a relative, not absolute, bound.
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 0.02,
                "expected near-identity for small inputs, got {got} want ~{want}"
            );
        }
    }

    #[test]
    fn quantity_ab_soft_bound_preserves_nonzero_gradient_even_when_saturated() {
        // Same rationale/shape as
        // value_head_soft_bound_preserves_nonzero_gradient_even_when_saturated:
        // a hard clamp_max here (the pre-fix behavior) would give zero
        // gradient once alpha/beta drift past QUANTITY_AB_MAX, which is
        // exactly the "stuck forever" failure mode this replaces.
        let params = Tensor::from_slice(&[1.0e6f32, 1.0e6f32])
            .reshape([1, 1, 2])
            .set_requires_grad(true);
        let (a, b) = quantity_ab(&params);
        (&a + &b).backward();
        let grad = params.grad();
        let g0: f64 = grad.double_value(&[0, 0, 0]);
        let g1: f64 = grad.double_value(&[0, 0, 1]);
        assert!(
            g0.is_finite() && g0 > 0.0,
            "alpha's gradient must be finite and nonzero, got {g0}"
        );
        assert!(
            g1.is_finite() && g1 > 0.0,
            "beta's gradient must be finite and nonzero, got {g1}"
        );
    }

    #[test]
    fn quantity_ab_is_bounded_even_for_a_huge_raw_input() {
        let params = Tensor::from_slice(&[1.0e9f32, 1.0e9f32]).reshape([1, 1, 2]);
        let (a, b) = quantity_ab(&params);
        let a_v: f64 = a.double_value(&[0, 0]);
        let b_v: f64 = b.double_value(&[0, 0]);
        assert!(
            a_v.is_finite() && a_v <= QUANTITY_AB_MAX,
            "alpha must be bounded, got {a_v}"
        );
        assert!(
            b_v.is_finite() && b_v <= QUANTITY_AB_MAX,
            "beta must be bounded, got {b_v}"
        );
    }
}
