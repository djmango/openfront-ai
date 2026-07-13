//! TS `UnitGrid` spatial index for `nearbyUnits` / `hasUnitNearby` parity.
//!
//! Iteration order within a cell/type `Set` is insertion order. Immobile
//! structures keep creation order, but **mobile** units (trade ships, warships,
//! boats, shells) that leave a cell and re-enter are re-`add`ed at the end of
//! the Set - native previously approximated this with a flat scan sorted by
//! `(cell_y, cell_x, type_idx, unit_id)`, which matches creation order only.
//! Equidistant warship target ties then pick different trade ships (found via
//! curriculum-parity-v4 `curr-b010-s2-pangaea` @14503).

use crate::execution::ordered_map::OrderedMap;
use crate::map::TileRef;
use std::collections::HashMap;

const CELL_SIZE: i32 = 100;

#[derive(Clone, Debug)]
struct UnitIndex {
    owner: u16,
    unit_type: String,
    tile: TileRef,
    cx: i32,
    cy: i32,
    under_construction: bool,
    active: bool,
}

#[derive(Clone, Debug, Default)]
pub struct UnitGrid {
    /// [cy][cx] -> unit_type -> ordered unit ids (TS `Map<UnitType, Set<Unit>>`).
    cells: Vec<Vec<HashMap<String, OrderedMap<i32, ()>>>>,
    index: HashMap<i32, UnitIndex>,
    width_cells: usize,
    height_cells: usize,
}

impl UnitGrid {
    pub fn new(map_width: u32, map_height: u32) -> Self {
        let width_cells = ((map_width as i32 + CELL_SIZE - 1) / CELL_SIZE).max(1) as usize;
        let height_cells = ((map_height as i32 + CELL_SIZE - 1) / CELL_SIZE).max(1) as usize;
        let cells = (0..height_cells)
            .map(|_| {
                (0..width_cells)
                    .map(|_| HashMap::new())
                    .collect()
            })
            .collect();
        Self {
            cells,
            index: HashMap::new(),
            width_cells,
            height_cells,
        }
    }

    fn cell_coords(x: u32, y: u32) -> (i32, i32) {
        (x as i32 / CELL_SIZE, y as i32 / CELL_SIZE)
    }

    fn valid_cell(&self, cx: i32, cy: i32) -> bool {
        cx >= 0 && cy >= 0 && (cx as usize) < self.width_cells && (cy as usize) < self.height_cells
    }

    pub fn add_unit(
        &mut self,
        owner: u16,
        unit_id: i32,
        unit_type: &str,
        tile: TileRef,
        x: u32,
        y: u32,
        under_construction: bool,
    ) {
        let (cx, cy) = Self::cell_coords(x, y);
        if !self.valid_cell(cx, cy) {
            return;
        }
        self.cells[cy as usize][cx as usize]
            .entry(unit_type.to_string())
            .or_default()
            .set(unit_id, ());
        self.index.insert(
            unit_id,
            UnitIndex {
                owner,
                unit_type: unit_type.to_string(),
                tile,
                cx,
                cy,
                under_construction,
                active: true,
            },
        );
    }

    pub fn remove_unit(&mut self, unit_id: i32) {
        let Some(entry) = self.index.remove(&unit_id) else {
            return;
        };
        if let Some(set) = self.cells[entry.cy as usize][entry.cx as usize].get_mut(&entry.unit_type)
        {
            set.remove(&unit_id);
        }
    }

    /// TS `UnitGrid.updateUnitCell` - re-inserts at end of the new cell's Set
    /// when the cell changes (matching JS Set delete+add).
    pub fn update_unit_tile(&mut self, unit_id: i32, tile: TileRef, x: u32, y: u32) {
        let Some(entry) = self.index.get(&unit_id).cloned() else {
            return;
        };
        let (cx, cy) = Self::cell_coords(x, y);
        if let Some(e) = self.index.get_mut(&unit_id) {
            e.tile = tile;
        }
        if entry.cx == cx && entry.cy == cy {
            return;
        }
        if self.valid_cell(entry.cx, entry.cy) {
            if let Some(set) = self.cells[entry.cy as usize][entry.cx as usize]
                .get_mut(&entry.unit_type)
            {
                set.remove(&unit_id);
            }
        }
        if let Some(e) = self.index.get_mut(&unit_id) {
            e.cx = cx;
            e.cy = cy;
        }
        if self.valid_cell(cx, cy) {
            self.cells[cy as usize][cx as usize]
                .entry(entry.unit_type)
                .or_default()
                .set(unit_id, ());
        }
    }

    pub fn set_owner(&mut self, unit_id: i32, owner: u16) {
        if let Some(entry) = self.index.get_mut(&unit_id) {
            entry.owner = owner;
        }
    }

    pub fn set_under_construction(&mut self, unit_id: i32, under_construction: bool) {
        if let Some(entry) = self.index.get_mut(&unit_id) {
            entry.under_construction = under_construction;
        }
    }

    pub fn set_active(&mut self, unit_id: i32, active: bool) {
        if let Some(entry) = self.index.get_mut(&unit_id) {
            entry.active = active;
        }
    }

    /// TS `UnitGrid.getCellsInRange` bounds (`Math.ceil` semantics).
    fn cells_in_range(
        &self,
        x: u32,
        y: u32,
        range: u32,
    ) -> (usize, usize, usize, usize) {
        let cell_size = CELL_SIZE;
        let (grid_x, grid_y) = Self::cell_coords(x, y);
        let x = x as i32;
        let y = y as i32;
        let range = range as i32;
        // JS `Math.ceil(n/d)`: for n>=0 use integer ceil; for n<0 Rust truncating
        // division toward 0 matches `Math.ceil` on negative non-integers.
        let ceil_div = |n: i32, d: i32| -> i32 {
            if n >= 0 {
                (n + d - 1) / d
            } else {
                n / d
            }
        };
        let start_grid_x =
            (grid_x - ceil_div(range - (x % cell_size), cell_size)).max(0) as usize;
        let end_grid_x = (grid_x + ceil_div(range - (cell_size - (x % cell_size)), cell_size))
            .min(self.width_cells as i32 - 1)
            .max(0) as usize;
        let start_grid_y =
            (grid_y - ceil_div(range - (y % cell_size), cell_size)).max(0) as usize;
        let end_grid_y = (grid_y + ceil_div(range - (cell_size - (y % cell_size)), cell_size))
            .min(self.height_cells as i32 - 1)
            .max(0) as usize;
        (start_grid_x, end_grid_x, start_grid_y, end_grid_y)
    }

    /// TS `UnitGrid.nearbyUnits(tile, range, types)` - cell row-major, then
    /// `types` array order, then Set insertion order within each cell/type.
    pub fn nearby_units(
        &self,
        tile_x: u32,
        tile_y: u32,
        range: u32,
        types: &[&str],
        include_under_construction: bool,
        tile_xy: impl Fn(TileRef) -> (u32, u32),
    ) -> Vec<(u16, i32, TileRef, f64)> {
        let (start_cx, end_cx, start_cy, end_cy) =
            self.cells_in_range(tile_x, tile_y, range);
        let range_sq = (range as i64) * (range as i64);
        let mut out = Vec::new();
        for cy in start_cy..=end_cy {
            for cx in start_cx..=end_cx {
                let cell = &self.cells[cy][cx];
                for &unit_type in types {
                    let Some(set) = cell.get(unit_type) else {
                        continue;
                    };
                    for (unit_id, _) in set.iter() {
                        let Some(entry) = self.index.get(&unit_id) else {
                            continue;
                        };
                        if !entry.active {
                            continue;
                        }
                        if !include_under_construction && entry.under_construction {
                            continue;
                        }
                        let (ux, uy) = tile_xy(entry.tile);
                        let dx = ux as i64 - tile_x as i64;
                        let dy = uy as i64 - tile_y as i64;
                        let d2 = dx * dx + dy * dy;
                        if d2 > range_sq {
                            continue;
                        }
                        out.push((entry.owner, unit_id, entry.tile, d2 as f64));
                    }
                }
            }
        }
        out
    }

    pub fn has_unit_nearby(
        &self,
        tile_x: u32,
        tile_y: u32,
        range: u32,
        unit_type: &str,
        owner_filter: Option<u16>,
        include_under_construction: bool,
        tile_xy: impl Fn(TileRef) -> (u32, u32),
    ) -> bool {
        self.nearby_units(
            tile_x,
            tile_y,
            range,
            &[unit_type],
            include_under_construction,
            tile_xy,
        )
        .into_iter()
        .any(|(owner, ..)| owner_filter.is_none_or(|o| o == owner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reentering_a_cell_moves_unit_to_end_of_set_order() {
        let mut grid = UnitGrid::new(300, 300);
        // Cell (0,0): tiles with x,y in 0..100
        grid.add_unit(1, 10, "Trade Ship", 5, 5, 5, false);
        grid.add_unit(1, 20, "Trade Ship", 6, 6, 6, false);
        let order: Vec<i32> = grid.nearby_units(5, 5, 50, &["Trade Ship"], false, |t| {
            (t as u32 % 300, t as u32 / 300)
        })
        .into_iter()
        .map(|(_, id, ..)| id)
        .collect();
        assert_eq!(order, vec![10, 20]);

        // Move unit 10 out of cell then back in - should go to end.
        grid.update_unit_tile(10, 150, 150, 5); // cell (1,0)
        grid.update_unit_tile(10, 7, 7, 7); // back to cell (0,0)
        let order: Vec<i32> = grid.nearby_units(5, 5, 50, &["Trade Ship"], false, |t| {
            (t as u32 % 300, t as u32 / 300)
        })
        .into_iter()
        .map(|(_, id, ..)| id)
        .collect();
        assert_eq!(order, vec![20, 10]);
    }
}
