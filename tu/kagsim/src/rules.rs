//! Turn processing. Ports `kaggriculture.py:298-509` and `:705-942`.

use crate::market::{process_market, refresh_prices, town_consume, Order};
use crate::rng::MtRandom;
use crate::state::*;

/// One farmer/hand op. Item and crop names stay as strings because validity is checked at apply
/// time in the reference (an unknown crop makes PLANT a silent no-op).
#[derive(Clone, Debug, Default)]
pub struct UnitAction {
    pub op: String,
    pub arg_s: Option<String>,
    pub arg_n: Option<i64>,
    /// A third element was present but not coercible to an int. `_apply_unit_action` does a bare
    /// `int(action[2])` for PICKUP and PLACE (`:347`, `:465`) with no try/except, unlike
    /// `_parse_order` (`:619`) — so the reference *raises* there. Recorded rather than silently
    /// defaulted, so the same input fails the same way.
    pub arg_n_bad: bool,
}

impl UnitAction {
    pub fn pass() -> Self {
        UnitAction { op: "PASS".into(), ..Default::default() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PlayerAction {
    pub farmer: UnitAction,
    pub hands: Vec<UnitAction>,
    pub market: Vec<Option<Order>>,
}

fn farmer_position(farm: &Farm, idx: usize) -> Option<(i32, i32)> {
    if idx == 0 {
        Some(farm.farmer)
    } else {
        farm.hands.get(idx - 1).copied()
    }
}

fn set_farmer_position(farm: &mut Farm, idx: usize, pos: (i32, i32)) {
    if idx == 0 {
        farm.farmer = pos;
    } else {
        farm.hands[idx - 1] = pos;
    }
}

/// `_apply_unit_action`, `:298`. Every invalid or illegal op is a silent no-op.
fn apply_unit_action(st: &mut GameState, p: usize, idx: usize, a: &UnitAction, day: i32) {
    if a.op.is_empty() {
        return;
    }
    let Some((fx, fy)) = farmer_position(&st.farms[p], idx) else { return };
    let bs = st.cfg.board_size;
    // `_farmer_inventory` is called before the op dispatch and grows the list (`:307`).
    st.privates[p].inventory_mut(idx);

    // Movement, `:309`. Out-of-bounds is a no-op; LOCKED tiles are passable (`:314`).
    let delta = match a.op.as_str() {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    };
    if let Some((dx, dy)) = delta {
        let (nx, ny) = (fx + dx, fy + dy);
        if nx >= 0 && nx < bs && ny >= 0 && ny < bs {
            set_farmer_position(&mut st.farms[p], idx, (nx, ny));
        }
        return;
    }
    if a.op == "PASS" {
        return;
    }

    let mut tile = st.tile(p, fx, fy);
    // 1.32.6 moved the shed operations ahead of the LOCKED guard (`:322-326`). They use the tile
    // only as a standing position -- the shed is always owned -- and three of the four shed-access
    // tiles start LOCKED, so the old ordering made them unreachable. PLACE is included: its animal
    // branch cannot match a LOCKED tile (kind is never a structure there), so it falls through to
    // the shed path exactly as the reference does.
    let shed_op = matches!(a.op.as_str(), "DROP" | "PICKUP" | "PLACE");
    if tile.kind == TileKind::Locked && !shed_op {
        return;
    }
    let shed_adj = is_shed_adjacent((fx, fy), bs);
    let cap = st.cfg.shed_capacity;

    match a.op.as_str() {
        // `:327`. Recomputes remaining room per item; overflow is discarded, and the entry is
        // removed from the inventory either way.
        "DROP" => {
            if !shed_adj {
                return;
            }
            let inv = std::mem::take(st.privates[p].inventory_mut(idx));
            for (item, n) in inv {
                if n <= 0 {
                    continue;
                }
                let room = (cap - st.privates[p].shed_total()).max(0);
                let take = n.min(room);
                if take > 0 {
                    st.privates[p].shed[item as usize] += take;
                }
                if st.collect_stats {
                    st.stats[p].discarded_overflow += n - take;
                }
            }
        }

        // `:342`. Seeds live in their own slot and are never picked up.
        "PICKUP" => {
            if !shed_adj {
                return;
            }
            let Some(name) = &a.arg_s else { return };
            if a.arg_n_bad {
                st.error = Some("invalid literal for int()".into());
                return;
            }
            let Some(item) = item_index(name) else { return };
            let mut n = a.arg_n.unwrap_or(1);
            if n <= 0 {
                return;
            }
            n = n.min(st.privates[p].shed[item]);
            if n <= 0 {
                return;
            }
            st.privates[p].shed[item] -= n;
            inv_add(st.privates[p].inventory_mut(idx), item, n);
        }

        "PLANT" => {
            let Some(name) = &a.arg_s else { return };
            let Some(crop) = crop_index(name) else { return };
            if tile.kind != TileKind::Empty {
                return;
            }
            if st.privates[p].seeds[crop] <= 0 {
                return;
            }
            st.privates[p].seeds[crop] -= 1;
            let t = Tile::new_plant(crop, day, st.cfg.turns_per_day);
            st.set_tile(p, fx, fy, t);
        }

        // `:375`. The yield bonus only accrues inside the window and only for one-time crops.
        "WATER" => {
            if tile.kind != TileKind::Plant || tile.watered_today {
                return;
            }
            tile.watered_today = true;
            let cd = &CROPS[tile.crop as usize];
            if !cd.ongoing {
                let age = day - tile.planted_day;
                let window_start = (cd.max_yield_day + 1) / 2;
                if age >= window_start && age <= cd.max_yield_day {
                    let bonus = if tile.fertilized_until_day >= day { 2 } else { 1 };
                    tile.yield_units = cd.max_yield.min(tile.yield_units + bonus);
                }
            }
            st.set_tile(p, fx, fy, tile);
        }

        // `:390`. The yield_units guard runs before the plant/animal split, so it also rejects
        // weeds and empty structures.
        "HARVEST" => {
            if matches!(tile.kind, TileKind::Empty | TileKind::Locked) {
                return;
            }
            if tile.yield_units <= 0 {
                return;
            }
            if tile.kind == TileKind::Plant {
                let cd = &CROPS[tile.crop as usize];
                if day - tile.planted_day < cd.first_yield_day {
                    return;
                }
                let units = tile.yield_units;
                tile.yield_units = 0;
                inv_add(st.privates[p].inventory_mut(idx), tile.crop as usize, units as i64);
                if cd.ongoing {
                    st.set_tile(p, fx, fy, tile);
                } else {
                    st.set_tile(p, fx, fy, Tile::EMPTY);
                }
            } else if let Some(an) = tile.animal {
                let units = tile.yield_units;
                tile.yield_units = 0;
                inv_add(st.privates[p].inventory_mut(idx), ANIMALS[an as usize].product, units as i64);
                st.set_tile(p, fx, fy, tile);
            }
        }

        // `:419`. Active for day, day+1, day+2.
        "FERTILIZE" => {
            if tile.kind != TileKind::Plant {
                return;
            }
            if !inv_take(st.privates[p].inventory_mut(idx), 8, 1) {
                return;
            }
            tile.fertilized_until_day = tile.fertilized_until_day.max(day + 2);
            st.set_tile(p, fx, fy, tile);
        }

        // `:428`. Removes plants, weeds, and *empty* structures; a placed animal blocks it.
        "DIG" => {
            if tile.kind == TileKind::Empty {
                return;
            }
            if tile.animal.is_some() {
                return;
            }
            st.set_tile(p, fx, fy, Tile::EMPTY);
        }

        "BUILD_COOP" | "BUILD_PASTURE" => {
            if tile.kind != TileKind::Empty {
                return;
            }
            let kind = if a.op == "BUILD_COOP" { TileKind::Coop } else { TileKind::Pasture };
            st.set_tile(p, fx, fy, Tile { kind, ..Tile::EMPTY });
        }

        // `:449`. Animal placement takes priority; otherwise it is a shed drop.
        "PLACE" => {
            let Some(name) = &a.arg_s else { return };
            if let Some(an) = animal_index(name) {
                let want = if ANIMALS[an].pasture { TileKind::Pasture } else { TileKind::Coop };
                if tile.kind == want && tile.animal.is_none() {
                    if inv_take(st.privates[p].inventory_mut(idx), item_index(name).unwrap(), 1) {
                        let t = Tile::new_animal(an, day);
                        st.set_tile(p, fx, fy, t);
                    }
                    return;
                }
            }
            if shed_adj {
                if a.arg_n_bad {
                    st.error = Some("invalid literal for int()".into());
                    return;
                }
                let Some(item) = item_index(name) else { return };
                let mut n = a.arg_n.unwrap_or(1);
                if n <= 0 {
                    return;
                }
                n = n.min(inv_get(st.privates[p].inventory_mut(idx), item));
                if n <= 0 {
                    return;
                }
                let room = (cap - st.privates[p].shed_total()).max(0);
                n = n.min(room);
                if n <= 0 {
                    return;
                }
                inv_take(st.privates[p].inventory_mut(idx), item, n);
                st.privates[p].shed[item] += n;
            }
        }

        "FEED" => {
            if tile.animal.is_none() || tile.watered_today {
                return;
            }
            if !inv_take(st.privates[p].inventory_mut(idx), 0, 1) {
                return; // needs 1 WHEAT
            }
            tile.watered_today = true;
            st.set_tile(p, fx, fy, tile);
        }

        "COLLECT_FERTILIZER" => {
            if tile.animal.is_none() || !tile.fertilizer_available {
                return;
            }
            tile.fertilizer_available = false;
            st.set_tile(p, fx, fy, tile);
            inv_add(st.privates[p].inventory_mut(idx), 8, 1);
        }

        "CARE" => {
            if tile.animal.is_none() || tile.cared_today {
                return;
            }
            tile.cared_today = true;
            st.set_tile(p, fx, fy, tile);
        }

        _ => {}
    }
}

/// Everything one unit action could possibly touch, for wasted-action accounting.
///
/// Detecting no-ops by fingerprinting rather than by threading a return value through every
/// branch keeps the ported rules byte-for-byte comparable with the reference — a diagnostic is
/// not worth risking a parity bug over.
type Fingerprint = ((i32, i32), Option<Tile>, Inv, [i64; N_ITEMS], [i64; N_CROPS]);

fn fingerprint(st: &GameState, p: usize, idx: usize) -> Fingerprint {
    let pos = farmer_position(&st.farms[p], idx).unwrap_or((-1, -1));
    let tile = if pos.0 >= 0 && pos.0 < st.cfg.board_size && pos.1 >= 0 && pos.1 < st.cfg.board_size
    {
        Some(st.tile(p, pos.0, pos.1))
    } else {
        None
    };
    let inv = st.privates[p].inventories.get(idx).cloned().unwrap_or_default();
    (pos, tile, inv, st.privates[p].shed, st.privates[p].seeds)
}

fn tile_eq(a: &Option<Tile>, b: &Option<Tile>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.kind == y.kind
                && x.crop == y.crop
                && x.animal == y.animal
                && x.planted_day == y.planted_day
                && x.yield_units == y.yield_units
                && x.max_lifespan_step == y.max_lifespan_step
                && x.fertilized_until_day == y.fertilized_until_day
                && x.consecutive_unwatered == y.consecutive_unwatered
                && x.pending_care_bonus == y.pending_care_bonus
                && x.watered_today == y.watered_today
                && x.cared_today == y.cared_today
                && x.fertilizer_available == y.fertilizer_available
        }
        _ => false,
    }
}

fn apply_and_count(st: &mut GameState, p: usize, idx: usize, a: &UnitAction, day: i32) {
    if !st.collect_stats {
        apply_unit_action(st, p, idx, a, day);
        return;
    }
    let before = fingerprint(st, p, idx);
    apply_unit_action(st, p, idx, a, day);
    let after = fingerprint(st, p, idx);

    st.stats[p].actions_total += 1;
    if before.0 != after.0 {
        st.stats[p].actions_move += 1;
    } else if tile_eq(&before.1, &after.1)
        && before.2 == after.2
        && before.3 == after.3
        && before.4 == after.4
    {
        st.stats[p].actions_noop += 1;
    }
}

/// `_decay_plants`, `:730`. Runs every step, decrementing every other step past the lifespan.
fn decay_plants(st: &mut GameState, p: usize, step: i32) {
    let n = st.farms[p].tiles.len();
    for i in 0..n {
        let t = st.farms[p].tiles[i];
        if t.kind != TileKind::Plant {
            continue;
        }
        let mls = t.max_lifespan_step;
        if mls < 0 || step < mls || (step - mls) % 2 != 0 {
            continue;
        }
        let mut t = t;
        t.yield_units -= 1;
        st.farms[p].tiles[i] = if t.yield_units <= 0 { Tile::WEED } else { t };
    }
}

/// `_daily_refresh_plants`, `:747`.
fn daily_refresh_plants(st: &mut GameState, p: usize, current_day: i32) {
    let next_day = current_day + 1;
    let tpd = st.cfg.turns_per_day;
    for i in 0..st.farms[p].tiles.len() {
        let mut t = st.farms[p].tiles[i];
        if t.kind != TileKind::Plant {
            continue;
        }
        let was_watered = t.watered_today;
        if was_watered {
            t.consecutive_unwatered = 0;
        } else {
            t.consecutive_unwatered += 1;
        }
        t.watered_today = false;
        if t.consecutive_unwatered >= 2 {
            st.farms[p].tiles[i] = Tile::WEED;
            continue;
        }
        let cd = &CROPS[t.crop as usize];
        if !cd.ongoing {
            st.farms[p].tiles[i] = t;
            continue;
        }
        let days_since_first = next_day - t.planted_day - cd.first_yield_day;
        if days_since_first < 0 || days_since_first % cd.interval != 0 {
            st.farms[p].tiles[i] = t;
            continue;
        }
        let production_count = days_since_first / cd.interval + 1;
        if production_count > cd.max_yield {
            st.farms[p].tiles[i] = t;
            continue;
        }
        // The fertilizer bonus only lands on a day the plant was also watered (`:777`).
        let fertilized = was_watered && t.fertilized_until_day >= current_day;
        t.yield_units = cd.max_yield.min(t.yield_units + if fertilized { 2 } else { 1 });
        if production_count == cd.max_yield {
            t.max_lifespan_step = (next_day + 1) * tpd;
        }
        st.farms[p].tiles[i] = t;
    }
}

/// `_daily_refresh_animals`, `:783`.
///
/// Ordering is load-bearing: feed check -> escape -> production (consuming the care bank only if
/// fed) -> care banking -> flag reset. So CARE banked on day d pays out on day d+1.
fn daily_refresh_animals(st: &mut GameState, p: usize, day: i32) {
    let next_day = day + 1;
    for i in 0..st.farms[p].tiles.len() {
        let mut t = st.farms[p].tiles[i];
        let Some(an) = t.animal else { continue };
        let a = &ANIMALS[an as usize];

        if t.watered_today {
            t.consecutive_unwatered = 0;
        } else {
            t.consecutive_unwatered += 1;
        }
        if t.consecutive_unwatered >= 2 {
            // The animal escapes; the bare structure remains.
            let kind = if a.pasture { TileKind::Pasture } else { TileKind::Coop };
            st.farms[p].tiles[i] = Tile { kind, ..Tile::EMPTY };
            continue;
        }
        let days_since_first = next_day - t.planted_day - a.first_yield_day;
        if days_since_first >= 0 && days_since_first % a.interval == 0 {
            let bonus = if t.watered_today { t.pending_care_bonus } else { 0 };
            t.yield_units = a.max_held.min(t.yield_units + 1 + bonus);
            t.pending_care_bonus = 0;
        }
        if t.cared_today && t.watered_today {
            t.pending_care_bonus += 1;
        }
        t.fertilizer_available = true;
        t.watered_today = false;
        t.cared_today = false;
        st.farms[p].tiles[i] = t;
    }
}

/// `_spawn_weeds`, `:814`. The RNG is only consumed for tiles that are actually empty — Python's
/// `and` short-circuits — so the draw count is gameplay-dependent.
fn spawn_weeds(st: &mut GameState, p: usize, rng: &mut MtRandom) {
    let bs = st.cfg.board_size;
    let chance = st.cfg.weed_spawn_chance;
    for y in 0..bs {
        for x in 0..bs {
            let i = (y * bs + x) as usize;
            if st.farms[p].tiles[i].kind == TileKind::Empty && rng.random() < chance {
                st.farms[p].tiles[i] = Tile::WEED;
            }
        }
    }
}

/// `_drop_inventories_to_shed`, `:821`. Insertion order decides who survives an overflow.
fn drop_inventories_to_shed(st: &mut GameState, p: usize) {
    let cap = st.cfg.shed_capacity;
    let invs = std::mem::take(&mut st.privates[p].inventories);
    for inv in invs {
        for (item, n) in inv {
            if n <= 0 {
                continue;
            }
            let room = (cap - st.privates[p].shed_total()).max(0);
            let take = n.min(room);
            if take > 0 {
                st.privates[p].shed[item as usize] += take;
            }
            if st.collect_stats {
                st.stats[p].discarded_overflow += n - take;
            }
        }
    }
    st.privates[p].inventories = vec![Inv::new()];
}

/// `_end_of_day`, `:838`.
fn end_of_day(st: &mut GameState, day: i32) {
    // Stable RNG keyed off the episode seed and the day so replays reproduce (`:849`).
    let mut rng = MtRandom::new((st.cfg.seed as i128 * 1_000_003) ^ day as i128);
    let bs = st.cfg.board_size;
    for p in 0..st.farms.len() {
        daily_refresh_plants(st, p, day);
        daily_refresh_animals(st, p, day);
        spawn_weeds(st, p, &mut rng);
        drop_inventories_to_shed(st, p);
        st.farms[p].farmer = default_spawn(bs);
        st.farms[p].hands.clear();
        st.farms[p].hires_today = 0;
    }
    let next_day = day + 1;
    let interval = st.cfg.town_shop_unlock_interval.max(1);
    if next_day > 0 && next_day % interval == 0 {
        // 1.32.6: drawn WITH replacement from all shops, capped at MAX_SHOP_INSTANCES = 8.
        // Previously each shop unlocked at most once, so by late game every product had a known
        // demand; now a product can have zero shops or four. SHOP_NAMES is already in the same
        // order as the reference's `sorted(SHOPS)`, so the RNG index maps directly.
        if st.unlocked_shops.len() < MAX_SHOP_INSTANCES {
            let pick = rng.choice_index(SHOP_NAMES.len() as u32) as usize;
            st.unlocked_shops.push(pick);
        }
    }
}

/// `interpreter`, `:871` — one turn.
pub fn step(st: &mut GameState, actions: &[PlayerAction]) {
    if st.done {
        return;
    }
    let tpd = st.cfg.turns_per_day.max(1);
    let step_no = st.step;
    let day = step_no / tpd;

    for p in 0..st.farms.len() {
        let a = &actions[p];

        // Atomic PLANT validation, `:897`: tally demand across farmer + hands *before* applying
        // anything; if demand for a crop exceeds the seeds on hand, drop every PLANT of it.
        let mut demand = [0i64; N_CROPS];
        let mut unknown_crop_requested = false;
        let mut tally = |u: &UnitAction| {
            if u.op == "PLANT" {
                match u.arg_s.as_deref().and_then(crop_index) {
                    Some(c) => demand[c] += 1,
                    None => {
                        if u.arg_s.is_some() {
                            unknown_crop_requested = true;
                        }
                    }
                }
            }
        };
        tally(&a.farmer);
        for h in &a.hands {
            tally(h);
        }
        let mut blocked = [false; N_CROPS];
        for c in 0..N_CROPS {
            blocked[c] = demand[c] > st.privates[p].seeds[c];
        }
        let _ = unknown_crop_requested; // an unknown crop is blocked and also a no-op anyway

        let allowed = |u: &UnitAction| -> UnitAction {
            if u.op == "PLANT" {
                if let Some(name) = &u.arg_s {
                    match crop_index(name) {
                        Some(c) if blocked[c] => return UnitAction::pass(),
                        None => return UnitAction::pass(), // demand > seeds.get(unknown, 0) == 0
                        _ => {}
                    }
                }
            }
            u.clone()
        };

        apply_and_count(st, p, 0, &allowed(&a.farmer), day);
        for (h, hand) in a.hands.iter().enumerate() {
            apply_and_count(st, p, h + 1, &allowed(hand), day);
        }
    }

    let queues: Vec<Vec<Option<Order>>> = actions.iter().map(|a| a.market.clone()).collect();
    process_market(st, &queues);
    town_consume(st, step_no);
    for p in 0..st.farms.len() {
        decay_plants(st, p, step_no);
    }
    if (step_no + 1) % tpd == 0 {
        end_of_day(st, day);
    }

    let next_step = step_no + 1;
    st.step = next_step;
    st.day = next_step / tpd;
    st.hour = next_step % tpd;

    // `:937` — fires DONE on the final recorded step.
    if step_no >= st.cfg.episode_steps - 2 {
        st.done = true;
    }
}

pub fn init(st: &mut GameState) {
    let params = st.cfg.market_params.clone();
    refresh_prices(&mut st.market, &params);
}
