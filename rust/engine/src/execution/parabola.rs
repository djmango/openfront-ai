//! Ballistic (cubic-Bezier) trajectory math for nukes (TS `PathFinder.Parabola.ts` +
//! `utilities/Line.ts` `CubicBezierCurve`/`DistanceBasedBezierCurve`).
//!
//! Only used for Atom Bomb / Hydrogen Bomb flight - MIRV Warheads take a degenerate
//! (zero-length, `src === dst`) curve in TS because `canSpawnUnitType(MIRVWarhead, ...)`
//! returns the target tile itself as the "spawn" tile, overwriting the execution's `src`
//! field before any curve is built. See `nuke_execution.rs` for the warhead fast path that
//! mirrors this by skipping curve construction entirely.

use crate::game::Game;
use crate::map::TileRef;

const PARABOLA_MIN_HEIGHT: f64 = 50.0;
const CURVE_PRECISION: f64 = 0.002;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pt {
    pub x: f64,
    pub y: f64,
}

fn within(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

fn point_at(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: f64) -> Pt {
    let tt_ = 1.0 - t;
    let tt = tt_ * tt_;
    let ttt = tt * tt_;
    let t2 = t * t;
    let t3 = t2 * t;
    Pt {
        x: ttt * p0.x + 3.0 * tt * t * p1.x + 3.0 * tt_ * t2 * p2.x + t3 * p3.x,
        y: ttt * p0.y + 3.0 * tt * t * p1.y + 3.0 * tt_ * t2 * p2.y + t3 * p3.y,
    }
}

fn compute_cached_points(p0: Pt, p1: Pt, p2: Pt, p3: Pt, pixel_spacing: f64) -> Vec<Pt> {
    let mut cached = Vec::new();
    let mut t = 0.0f64;
    let mut prev = point_at(p0, p1, p2, p3, t);
    cached.push(prev);
    let mut cumulative = 0.0f64;
    while t < 1.0 {
        t = (t + CURVE_PRECISION).min(1.0);
        let current = point_at(p0, p1, p2, p3, t);
        let dx = current.x - prev.x;
        let dy = current.y - prev.y;
        cumulative += (dx * dx + dy * dy).sqrt();
        if cumulative >= pixel_spacing {
            cached.push(current);
            cumulative = 0.0;
        }
        prev = current;
    }
    let final_point = point_at(p0, p1, p2, p3, 1.0);
    let need_push = match cached.last() {
        None => true,
        Some(last) => *last != final_point,
    };
    if need_push {
        cached.push(final_point);
    }
    cached
}

/// TS `DistanceBasedBezierCurve` - precomputed points + cumulative-distance stepping.
pub struct Curve {
    points: Vec<Pt>,
    cum_dist: Vec<f64>,
    total_distance: f64,
    current_index: usize,
}

impl Curve {
    fn new(p0: Pt, p1: Pt, p2: Pt, p3: Pt, increment: f64) -> Self {
        let points = compute_cached_points(p0, p1, p2, p3, increment);
        let mut cum_dist = vec![0.0; points.len()];
        for i in 1..points.len() {
            let dx = points[i].x - points[i - 1].x;
            let dy = points[i].y - points[i - 1].y;
            cum_dist[i] = cum_dist[i - 1] + (dx * dx + dy * dy).sqrt();
        }
        Self {
            points,
            cum_dist,
            total_distance: 0.0,
            current_index: 0,
        }
    }

    pub fn all_points(&self) -> &[Pt] {
        &self.points
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// TS `DistanceBasedBezierCurve.increment(distance)`.
    pub fn increment(&mut self, distance: f64) -> Option<Pt> {
        self.total_distance += distance;
        while self.current_index < self.points.len() - 1
            && self.cum_dist[self.current_index + 1] < self.total_distance
        {
            self.current_index += 1;
        }
        if self.current_index >= self.points.len() - 1 {
            return None;
        }
        Some(self.points[self.current_index])
    }
}

/// TS `ParabolaUniversalPathFinder.createCurve(from, to)`.
pub fn create_curve(
    game: &Game,
    from: TileRef,
    to: TileRef,
    increment: f64,
    distance_based_height: bool,
    direction_up: bool,
) -> Curve {
    let p0 = Pt {
        x: game.x(from) as f64,
        y: game.y(from) as f64,
    };
    let p3 = Pt {
        x: game.x(to) as f64,
        y: game.y(to) as f64,
    };
    let dx = p3.x - p0.x;
    let dy = p3.y - p0.y;
    let distance = (dx * dx + dy * dy).sqrt();
    let max_height = if distance_based_height {
        (distance / 3.0).max(PARABOLA_MIN_HEIGHT)
    } else {
        0.0
    };
    let height_mult = if direction_up { -1.0 } else { 1.0 };
    let map_height = game.height() as f64;
    let p1 = Pt {
        x: p0.x + dx / 4.0,
        y: within(p0.y + dy / 4.0 + height_mult * max_height, 0.0, map_height - 1.0),
    };
    let p2 = Pt {
        x: p0.x + dx * 3.0 / 4.0,
        y: within(
            p0.y + dy * 3.0 / 4.0 + height_mult * max_height,
            0.0,
            map_height - 1.0,
        ),
    };
    Curve::new(p0, p1, p2, p3, increment)
}

pub fn point_to_tile(game: &Game, p: Pt) -> TileRef {
    game.ref_xy(p.x.floor() as u32, p.y.floor() as u32)
}

/// TS `findPath` - all cached curve points floored to tiles (used for the precomputed
/// `trajectory` array).
pub fn find_path_tiles(
    game: &Game,
    from: TileRef,
    to: TileRef,
    increment: f64,
    distance_based_height: bool,
    direction_up: bool,
) -> Vec<TileRef> {
    let curve = create_curve(game, from, to, increment, distance_based_height, direction_up);
    curve
        .all_points()
        .iter()
        .map(|p| point_to_tile(game, *p))
        .collect()
}

#[cfg(test)]
mod tests {
    /// TS `openfront/tests/NukeTrajectory.test.ts` exercises
    /// `src/client/render/gl/utils/NukeTrajectory.ts`'s `computeNukeControlPoints`
    /// / `computeTrajectoryThresholds` / `buildNukeTrajectory` - pure
    /// client-side GL-renderer math (Bezier control points plus t-value
    /// color thresholds for drawing the on-screen nuke arc: untargetable
    /// mid-air zone, SAM-intercept point, impassable-terrain-block point).
    /// `rust/engine` is the headless simulation engine and has no rendering
    /// layer at all (no other module in this crate even mentions "Bezier"
    /// besides this file) - there is nothing in `rust/engine/src` for that
    /// TS file's actual subject (a `computeTrajectoryThresholds`-equivalent
    /// producing UI color-threshold t-values) to test, so it is not ported.
    ///
    /// This file's own `create_curve`/`find_path_tiles` already port the
    /// *simulation-relevant* half of the same underlying math (TS
    /// `ParabolaUniversalPathFinder.createCurve`/`findPath`, i.e. the exact
    /// control-point formula `computeNukeControlPoints` mirrors for
    /// rendering), and the specific simulation behavior the TS test's
    /// "impassable terrain blocks the trajectory" scenarios are motivated
    /// by - a nuke launch aborting when its flight path crosses impassable
    /// terrain - is already fully covered by
    /// `nuke_execution.rs::impassable_terrain_tests` (`can_build_atom_bomb_returns_none_for_impassable_target`,
    /// `nuke_trajectory_blocked_by_impassable_terrain_aborts_launch`,
    /// `nuke_can_launch_when_trajectory_does_not_cross_impassable_terrain`,
    /// etc.) and `nuke_ai.rs::is_trajectory_blocked_by_impassable`. Kept as
    /// a documented, permanently-ignored marker (this codebase's precedent
    /// for "confirmed out of scope", e.g. `warship_ai.rs`'s
    /// `findTeamGameWarshipTarget` note) rather than silently dropping the
    /// file from the porting pass.
    #[test]
    #[ignore = "NukeTrajectory.test.ts targets client-only GL-renderer math (src/client/render/gl/utils/NukeTrajectory.ts) with no rust/engine simulation equivalent; see doc comment"]
    fn nuke_trajectory_test_ts_targets_client_only_renderer_math_not_the_simulation() {}
}
