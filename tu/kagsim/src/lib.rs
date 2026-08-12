//! Fast Kaggriculture simulator.
//!
//! Parity target: `kaggle_environments/envs/kaggriculture/kaggriculture.py`.
//! Ported rules cite the line they came from; see `TASKS.md` T0.2 for the checklist of behaviours
//! that must match exactly, and `tests/test_parity.py` for the harness that proves they do.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub mod market;
pub mod rng;
pub mod rules;
pub mod state;

use rules::{PlayerAction, UnitAction};
use state::*;

/// Round half to even, matching Python's `round()`.
///
/// `kaggriculture.py:192` does `int(round(price))`. Python 3 uses banker's rounding, so
/// `round(2.5) == 2`, whereas Rust's `f64::round()` rounds half away from zero and gives 3.
/// Every place the reference calls `round` must go through this.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

#[pyfunction]
#[pyo3(name = "round_half_even")]
fn py_round_half_even(x: f64) -> f64 {
    round_half_even(x)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python `int(x)` semantics, returning None where `int()` would raise.
///
/// Both `_apply_unit_action` (`:347`) and `_parse_order` (`:619`) coerce their quantity with a
/// bare `int()`, which accepts ints, floats (truncating toward zero) and numeric strings. A plain
/// `extract::<i64>()` rejects the last two, which would silently drop orders the reference honours.
fn py_int(v: &Bound<'_, PyAny>) -> Option<i64> {
    if let Ok(n) = v.extract::<i64>() {
        return Some(n);
    }
    if let Ok(f) = v.extract::<f64>() {
        return if f.is_finite() { Some(f.trunc() as i64) } else { None };
    }
    if let Ok(s) = v.extract::<String>() {
        // `int("5")` and `int(" 7 ")` work; `int("5.5")` raises.
        return s.trim().parse::<i64>().ok();
    }
    None
}

fn parse_unit_action(obj: &Bound<'_, PyAny>) -> UnitAction {
    let Ok(list) = obj.cast::<PyList>() else {
        return UnitAction::default();
    };
    if list.is_empty() {
        return UnitAction::default();
    }
    let op = list.get_item(0).ok().and_then(|v| v.extract::<String>().ok()).unwrap_or_default();
    let arg_s = if list.len() >= 2 {
        list.get_item(1).ok().and_then(|v| v.extract::<String>().ok())
    } else {
        None
    };
    let (arg_n, arg_n_bad) = if list.len() >= 3 {
        match list.get_item(2).ok().as_ref().and_then(py_int) {
            Some(n) => (Some(n), false),
            None => (None, true),
        }
    } else {
        (None, false)
    };
    UnitAction { op, arg_s, arg_n, arg_n_bad }
}

fn parse_player_action(obj: &Bound<'_, PyAny>) -> PlayerAction {
    let mut pa = PlayerAction { farmer: UnitAction::pass(), ..Default::default() };
    let Ok(d) = obj.cast::<PyDict>() else { return pa };

    if let Ok(Some(f)) = d.get_item("farmer") {
        pa.farmer = parse_unit_action(&f);
    }
    if let Ok(Some(h)) = d.get_item("hands") {
        if let Ok(list) = h.cast::<PyList>() {
            for i in 0..list.len() {
                if let Ok(item) = list.get_item(i) {
                    pa.hands.push(parse_unit_action(&item));
                }
            }
        }
    }
    if let Ok(Some(m)) = d.get_item("market") {
        if let Ok(list) = m.cast::<PyList>() {
            for i in 0..list.len() {
                // Malformed orders still occupy their slot (`_parse_order` returns None in
                // place), so push None rather than skipping.
                let Ok(item) = list.get_item(i) else { pa.market.push(None); continue };
                let Ok(o) = item.cast::<PyList>() else { pa.market.push(None); continue };
                if o.is_empty() {
                    pa.market.push(None);
                    continue;
                }
                let op: String = match o.get_item(0).ok().and_then(|v| v.extract().ok()) {
                    Some(s) => s,
                    None => { pa.market.push(None); continue }
                };
                let item_name: Option<String> =
                    if o.len() >= 2 { o.get_item(1).ok().and_then(|v| v.extract().ok()) } else { None };
                let n: Option<i64> =
                    if o.len() >= 3 { o.get_item(2).ok().as_ref().and_then(py_int) } else { None };
                pa.market.push(market::parse_order(&op, item_name.as_deref(), n));
            }
        }
    }
    pa
}

fn tile_canonical<'py>(py: Python<'py>, t: &Tile) -> PyResult<Bound<'py, PyAny>> {
    // Shapes chosen so an *empty* structure (`{"kind": "COOP"}` with no other keys) can never be
    // confused with an occupied one.
    let items: Vec<Py<PyAny>> = match t.kind {
        TileKind::Empty => vec!["EMPTY".into_pyobject(py)?.into()],
        TileKind::Locked => vec!["LOCKED".into_pyobject(py)?.into()],
        TileKind::Weed => vec!["WEED".into_pyobject(py)?.into()],
        TileKind::Plant => vec![
            "PLANT".into_pyobject(py)?.into(),
            CROP_NAMES[t.crop as usize].into_pyobject(py)?.into(),
            t.planted_day.into_pyobject(py)?.into(),
            t.watered_today.into_pyobject(py)?.to_owned().into(),
            t.consecutive_unwatered.into_pyobject(py)?.into(),
            t.yield_units.into_pyobject(py)?.into(),
            t.max_lifespan_step.into_pyobject(py)?.into(),
            t.fertilized_until_day.into_pyobject(py)?.into(),
        ],
        TileKind::Coop | TileKind::Pasture => {
            let base = if t.kind == TileKind::Coop { "COOP" } else { "PASTURE" };
            match t.animal {
                None => vec![base.into_pyobject(py)?.into()],
                Some(an) => vec![
                    format!("{base}_A").into_pyobject(py)?.into(),
                    ANIMAL_NAMES[an as usize].into_pyobject(py)?.into(),
                    t.planted_day.into_pyobject(py)?.into(),
                    t.yield_units.into_pyobject(py)?.into(),
                    t.consecutive_unwatered.into_pyobject(py)?.into(),
                    t.watered_today.into_pyobject(py)?.to_owned().into(),
                    t.cared_today.into_pyobject(py)?.to_owned().into(),
                    t.fertilizer_available.into_pyobject(py)?.to_owned().into(),
                    t.pending_care_bonus.into_pyobject(py)?.into(),
                ],
            }
        }
    };
    Ok(PyList::new(py, items)?.into_any())
}

/// The exact tile shape agents see. An empty structure is literally `{"kind": "COOP"}` with no
/// other keys (`:440`), which some agent code distinguishes with `"animal" in tile`.
fn tile_observation<'py>(py: Python<'py>, t: &Tile) -> PyResult<Bound<'py, PyAny>> {
    match t.kind {
        TileKind::Empty => Ok(py.None().into_bound(py)),
        TileKind::Locked => Ok("LOCKED".into_pyobject(py)?.into_any()),
        TileKind::Weed => {
            let d = PyDict::new(py);
            d.set_item("kind", "WEED")?;
            Ok(d.into_any())
        }
        TileKind::Plant => {
            let d = PyDict::new(py);
            d.set_item("kind", "PLANT")?;
            d.set_item("crop", CROP_NAMES[t.crop as usize])?;
            d.set_item("planted_day", t.planted_day)?;
            d.set_item("watered_today", t.watered_today)?;
            d.set_item("consecutive_unwatered", t.consecutive_unwatered)?;
            d.set_item("yield_units", t.yield_units)?;
            d.set_item("max_lifespan_step", t.max_lifespan_step)?;
            d.set_item("fertilized_until_day", t.fertilized_until_day)?;
            Ok(d.into_any())
        }
        TileKind::Coop | TileKind::Pasture => {
            let kind = if t.kind == TileKind::Coop { "COOP" } else { "PASTURE" };
            let d = PyDict::new(py);
            d.set_item("kind", kind)?;
            if let Some(an) = t.animal {
                d.set_item("animal", ANIMAL_NAMES[an as usize])?;
                d.set_item("placed_day", t.planted_day)?;
                d.set_item("yield_units", t.yield_units)?;
                d.set_item("fed_today", t.watered_today)?;
                d.set_item("consecutive_unfed", t.consecutive_unwatered)?;
                d.set_item("cared_today", t.cared_today)?;
                d.set_item("fertilizer_available", t.fertilizer_available)?;
                d.set_item("pending_care_bonus", t.pending_care_bonus)?;
            }
            Ok(d.into_any())
        }
    }
}

#[pyclass]
pub struct Sim {
    st: GameState,
}

#[pymethods]
impl Sim {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut cfg = Config::default();
        if let Some(d) = config {
            macro_rules! get {
                ($k:literal, $f:ident, $t:ty) => {
                    if let Ok(Some(v)) = d.get_item($k) {
                        cfg.$f = v.extract::<$t>()?;
                    }
                };
            }
            get!("episodeSteps", episode_steps, i32);
            get!("boardSize", board_size, i32);
            get!("startingMoney", starting_money, i64);
            get!("maxMarketOrdersPerTurn", max_market_orders, usize);
            get!("turnsPerDay", turns_per_day, i32);
            get!("shedCapacity", shed_capacity, i64);
            get!("weedSpawnChance", weed_spawn_chance, f64);
            get!("townShopUnlockInterval", town_shop_unlock_interval, i32);
            get!("townShopSellInterval", town_shop_sell_interval, i32);
            get!("townCenterSellInterval", town_center_sell_interval, i32);
            get!("farmHandCostMult", farm_hand_cost_mult, i64);
            get!("seed", seed, i64);
            // `marketParams`, `:242`. Sparse per-product overrides merged onto the defaults.
            // The reference treats any *truthy* value as "overridden" and then stores the
            // resolved table in `market["params"]`, which is observable state — so the flag has
            // to follow truthiness, not whether any key actually matched a product.
            if let Ok(Some(v)) = d.get_item("marketParams") {
                let Ok(m) = v.cast::<PyDict>() else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "marketParams must be a dict of {product: {field: value}}",
                    ));
                };
                if !m.is_empty() {
                    cfg.market_params_overridden = true;
                    for (key, patch) in m.iter() {
                        let Ok(name) = key.extract::<String>() else { continue };
                        // Unknown product names and non-dict patches are ignored (`:69`).
                        let Some(idx) = item_index(&name).filter(|&i| i < N_PRODUCTS) else {
                            continue;
                        };
                        let Ok(pd) = patch.cast::<PyDict>() else { continue };
                        let mp = &mut cfg.market_params[idx];
                        if let Ok(Some(x)) = pd.get_item("base") {
                            mp.base = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("I0") {
                            mp.i0 = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("T") {
                            mp.t = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("below_func") {
                            mp.below_func = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("below_target") {
                            mp.below_target = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("above_func") {
                            mp.above_func = x.extract()?;
                        }
                        if let Ok(Some(x)) = pd.get_item("above_target") {
                            mp.above_target = x.extract()?;
                        }
                        mp.resync();
                    }
                }
            }
        }
        let mut st = GameState::new(cfg);
        rules::init(&mut st);
        Ok(Sim { st })
    }

    #[getter]
    fn step_no(&self) -> i32 {
        self.st.step
    }
    #[getter]
    fn done(&self) -> bool {
        self.st.done
    }

    fn money(&self, player: usize) -> f64 {
        self.st.farms[player].money
    }

    /// Enable wasted-action / overflow / sales accounting. Off by default — it fingerprints state
    /// around every unit action, which is pure overhead for training rollouts.
    #[setter]
    fn set_collect_stats(&mut self, on: bool) {
        self.st.collect_stats = on;
    }
    #[getter]
    fn collect_stats(&self) -> bool {
        self.st.collect_stats
    }

    fn stats<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.st.stats[player];
        let d = PyDict::new(py);
        d.set_item("actions_total", s.actions_total)?;
        d.set_item("actions_noop", s.actions_noop)?;
        d.set_item("actions_move", s.actions_move)?;
        d.set_item("actions_effective", s.actions_total - s.actions_noop - s.actions_move)?;
        d.set_item("discarded_overflow", s.discarded_overflow)?;
        let units = PyDict::new(py);
        let revenue = PyDict::new(py);
        for i in 0..N_PRODUCTS {
            if s.sold_units[i] != 0 {
                units.set_item(ITEM_NAMES[i], s.sold_units[i])?;
                revenue.set_item(ITEM_NAMES[i], s.sold_revenue[i])?;
            }
        }
        d.set_item("sold_units", units)?;
        d.set_item("sold_revenue", revenue)?;
        Ok(d)
    }

    /// Apply one turn. `actions` is a list of per-player dicts in the Kaggle action format.
    fn step(&mut self, actions: &Bound<'_, PyList>) -> PyResult<()> {
        let mut parsed: Vec<PlayerAction> = Vec::with_capacity(self.st.farms.len());
        for i in 0..self.st.farms.len() {
            parsed.push(match actions.get_item(i) {
                Ok(a) => parse_player_action(&a),
                Err(_) => PlayerAction { farmer: UnitAction::pass(), ..Default::default() },
            });
        }
        rules::step(&mut self.st, &parsed);
        if let Some(msg) = self.st.error.take() {
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        Ok(())
    }

    /// Build the Kaggle-format observation for `player`, so agents written against the real
    /// environment run unmodified on top of kagsim.
    fn observation<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        let bs = self.st.cfg.board_size;
        let out = PyDict::new(py);
        out.set_item("player", player)?;
        // `step` reaches BOTH seats, correct on every turn of a 719-turn episode (measured).
        //
        // This previously emitted `step` for player 0 only, to mirror what the *stored* replay
        // state shows: `env.steps[i][1]["observation"]` really does lack the key. But that is not
        // the surface an agent sees. Agents are handed `Environment.__get_shared_state(position)`
        // (`core.py:725-736`), which reconstructs the observation and carries `step` to both
        // seats. The old parity test compared against the stored state and so confirmed the
        // omission rather than detecting it.
        out.set_item("step", self.st.step)?;
        out.set_item("day", self.st.day)?;
        out.set_item("hour", self.st.hour)?;

        let farms = PyList::empty(py);
        for f in &self.st.farms {
            let d = PyDict::new(py);
            d.set_item("money", f.money)?;
            let rows = PyList::empty(py);
            for y in 0..bs {
                let row = PyList::empty(py);
                for x in 0..bs {
                    row.append(tile_observation(py, &f.tiles[(y * bs + x) as usize])?)?;
                }
                rows.append(row)?;
            }
            d.set_item("tiles", rows)?;
            d.set_item("farmer", vec![f.farmer.0, f.farmer.1])?;
            d.set_item("hands", f.hands.iter().map(|h| vec![h.0, h.1]).collect::<Vec<_>>())?;
            let mut quads = vec!["NW".to_string()];
            for &q in LAND_ORDER.iter() {
                if f.unlocked[q as usize] {
                    quads.push(QUADRANT_NAMES[q as usize].to_string());
                }
            }
            d.set_item("unlocked_quadrants", quads)?;
            d.set_item("hires_today", f.hires_today)?;
            farms.append(d)?;
        }
        out.set_item("farms", farms)?;

        let market = PyDict::new(py);
        let inv = PyDict::new(py);
        let prices = PyDict::new(py);
        for i in 0..N_PRODUCTS {
            inv.set_item(ITEM_NAMES[i], self.st.market.inventory[i])?;
            prices.set_item(ITEM_NAMES[i], self.st.market.prices[i])?;
        }
        market.set_item("inventory", inv)?;
        market.set_item("prices", prices)?;
        if self.st.cfg.market_params_overridden {
            market.set_item("params", market_params_dict(py, &self.st.cfg.market_params)?)?;
        }
        out.set_item("market", market)?;

        let town = PyDict::new(py);
        town.set_item(
            "unlocked_shops",
            self.st.unlocked_shops.iter().map(|&s| SHOP_NAMES[s]).collect::<Vec<_>>(),
        )?;
        out.set_item("town", town)?;

        // The reference pre-populates every shed and seed key (`:157`), so keep zeros here —
        // agents index these dicts directly.
        let pv = &self.st.privates[player];
        let private = PyDict::new(py);
        let shed = PyDict::new(py);
        for i in 0..N_ITEMS {
            shed.set_item(ITEM_NAMES[i], pv.shed[i])?;
        }
        private.set_item("shed", shed)?;
        let seeds = PyDict::new(py);
        for i in 0..N_CROPS {
            seeds.set_item(CROP_NAMES[i], pv.seeds[i])?;
        }
        private.set_item("seeds", seeds)?;
        let invs = PyList::empty(py);
        for inv in &pv.inventories {
            let d = PyDict::new(py);
            for (item, n) in inv {
                d.set_item(ITEM_NAMES[*item as usize], n)?;
            }
            invs.append(d)?;
        }
        private.set_item("inventories", invs)?;
        out.set_item("private", private)?;
        Ok(out)
    }

    /// Ordering-stable snapshot for parity comparison. Zero-valued shed/seed entries are dropped
    /// so key presence can't differ from the reference's dicts.
    fn canonical_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("step", self.st.step)?;
        out.set_item("day", self.st.day)?;
        out.set_item("hour", self.st.hour)?;
        out.set_item(
            "shops",
            self.st.unlocked_shops.iter().map(|&s| SHOP_NAMES[s]).collect::<Vec<_>>(),
        )?;

        let minv = PyDict::new(py);
        let mpr = PyDict::new(py);
        for i in 0..N_PRODUCTS {
            minv.set_item(ITEM_NAMES[i], self.st.market.inventory[i])?;
            mpr.set_item(ITEM_NAMES[i], self.st.market.prices[i])?;
        }
        out.set_item("market_inv", minv)?;
        out.set_item("market_prices", mpr)?;
        if self.st.cfg.market_params_overridden {
            out.set_item("market_params", market_params_dict(py, &self.st.cfg.market_params)?)?;
        }

        let farms = PyList::empty(py);
        for f in &self.st.farms {
            let d = PyDict::new(py);
            d.set_item("money", f.money)?;
            d.set_item("farmer", vec![f.farmer.0, f.farmer.1])?;
            d.set_item(
                "hands",
                f.hands.iter().map(|h| vec![h.0, h.1]).collect::<Vec<_>>(),
            )?;
            let mut quads = vec!["NW".to_string()];
            for &q in LAND_ORDER.iter() {
                if f.unlocked[q as usize] {
                    quads.push(QUADRANT_NAMES[q as usize].to_string());
                }
            }
            d.set_item("unlocked", quads)?;
            d.set_item("hires_today", f.hires_today)?;
            let tiles = PyList::empty(py);
            for t in &f.tiles {
                tiles.append(tile_canonical(py, t)?)?;
            }
            d.set_item("tiles", tiles)?;
            farms.append(d)?;
        }
        out.set_item("farms", farms)?;

        let privs = PyList::empty(py);
        for pv in &self.st.privates {
            let d = PyDict::new(py);
            let shed = PyDict::new(py);
            for i in 0..N_ITEMS {
                if pv.shed[i] != 0 {
                    shed.set_item(ITEM_NAMES[i], pv.shed[i])?;
                }
            }
            d.set_item("shed", shed)?;
            let seeds = PyDict::new(py);
            for i in 0..N_CROPS {
                if pv.seeds[i] != 0 {
                    seeds.set_item(CROP_NAMES[i], pv.seeds[i])?;
                }
            }
            d.set_item("seeds", seeds)?;
            let invs = PyList::empty(py);
            for inv in &pv.inventories {
                let l = PyList::empty(py);
                for (item, n) in inv {
                    l.append(vec![
                        ITEM_NAMES[*item as usize].into_pyobject(py)?.into_any().unbind(),
                        n.into_pyobject(py)?.into_any().unbind(),
                    ])?;
                }
                invs.append(l)?;
            }
            d.set_item("inventories", invs)?;
            privs.append(d)?;
        }
        out.set_item("privates", privs)?;
        Ok(out)
    }
}

/// Price lookup for agent code. `params` accepts an `obs["market"]["params"]` table; omitting it
/// uses the defaults, which is only correct when the episode has no `marketParams` override.
#[pyfunction]
#[pyo3(signature = (item, inventory, params = None))]
fn market_price(item: usize, inventory: i64, params: Option<&Bound<'_, PyDict>>) -> PyResult<i64> {
    let table = match params {
        None => market::default_market_params(),
        Some(d) => parse_market_params(d)?,
    };
    Ok(market::market_price(item, inventory, &table))
}

/// Build a full 9-entry table from a `market["params"]`-shaped dict.
fn parse_market_params(d: &Bound<'_, PyDict>) -> PyResult<Vec<market::MarketParam>> {
    let mut table = market::default_market_params();
    for (key, patch) in d.iter() {
        let Ok(name) = key.extract::<String>() else { continue };
        let Some(idx) = item_index(&name).filter(|&i| i < N_PRODUCTS) else { continue };
        let Ok(pd) = patch.cast::<PyDict>() else { continue };
        let mp = &mut table[idx];
        if let Ok(Some(x)) = pd.get_item("base") { mp.base = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("I0") { mp.i0 = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("T") { mp.t = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("below_func") { mp.below_func = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("below_target") { mp.below_target = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("above_func") { mp.above_func = x.extract()?; }
        if let Ok(Some(x)) = pd.get_item("above_target") { mp.above_target = x.extract()?; }
        mp.resync();
    }
    Ok(table)
}

/// The resolved table as the reference exposes it in `market["params"]`.
fn market_params_dict<'py>(py: Python<'py>, table: &[market::MarketParam]) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (i, p) in table.iter().enumerate() {
        let d = PyDict::new(py);
        d.set_item("base", p.base)?;
        d.set_item("I0", p.i0)?;
        d.set_item("T", p.t)?;
        d.set_item("below_func", &p.below_func)?;
        d.set_item("below_target", p.below_target)?;
        d.set_item("above_func", &p.above_func)?;
        d.set_item("above_target", p.above_target)?;
        out.set_item(ITEM_NAMES[i], d)?;
    }
    Ok(out)
}

#[pymodule]
fn kagsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(py_round_half_even, m)?)?;
    m.add_function(wrap_pyfunction!(market_price, m)?)?;
    m.add_class::<Sim>()?;
    rng::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banker_rounding_matches_python() {
        // Verified against CPython: round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4.
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-1.5), -2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(2.4), 2.0);
        assert_eq!(round_half_even(2.6), 3.0);
    }

    #[test]
    fn fib_is_one_indexed_from_one() {
        // `_hire_cost` uses fib(hires_today): 1, 1, 2, 3, 5, 8, 13, 21.
        let got: Vec<i64> = (0..8).map(fib).collect();
        assert_eq!(got, vec![1, 1, 2, 3, 5, 8, 13, 21]);
    }

    #[test]
    fn new_plant_starts_unwatered_and_with_one_unit() {
        let t = Tile::new_plant(0, 0, 24); // WHEAT
        assert_eq!(t.consecutive_unwatered, 1, "planting day counts as already missed");
        assert_eq!(t.yield_units, 1, "one-time crops start harvestable at 1");
        assert_eq!(t.max_lifespan_step, (0 + 4 + 1) * 24);
        let t = Tile::new_plant(2, 0, 24); // TOMATO, ongoing
        assert_eq!(t.yield_units, 0);
        assert_eq!(t.max_lifespan_step, -1);
    }

    #[test]
    fn water_bonus_window_bounds() {
        // window_start = (max_yield_day + 1) // 2
        assert_eq!((CROPS[0].max_yield_day + 1) / 2, 2); // WHEAT: ages 2..4
        assert_eq!((CROPS[1].max_yield_day + 1) / 2, 2); // CARROT: ages 2..3
        assert_eq!((CROPS[4].max_yield_day + 1) / 2, 6); // MELON: ages 6..12
    }

    #[test]
    fn inventory_preserves_insertion_order_and_reinsertion() {
        let mut inv = Inv::new();
        inv_add(&mut inv, 5, 2);
        inv_add(&mut inv, 0, 1);
        assert_eq!(inv.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![5, 0]);
        // Draining to zero deletes the key; re-adding appends at the end, as Python dicts do.
        assert!(inv_take(&mut inv, 5, 2));
        inv_add(&mut inv, 5, 1);
        assert_eq!(inv.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 5]);
        assert!(!inv_take(&mut inv, 0, 99), "cannot take more than held");
    }

    #[test]
    fn market_price_floor_and_base() {
        let t = market::default_market_params();
        let i0 = market::DEFAULT_I0;
        assert_eq!(market::market_price(0, i0, &t), 25); // WHEAT base
        assert_eq!(market::market_price(4, i0, &t), 250); // MELON base
        // MELON: sq shape, above_target 3.6 -> floors well before T = 300.
        assert_eq!(market::market_price(4, i0 + 300, &t), 1);
        assert!(market::market_price(5, i0 + 3000, &t) > 30); // EGG is a deep sink

        // A sparse override changes only what it names.
        let mut o = market::default_market_params();
        o[0] = market::MarketParam::new(25.0, i0, 400.0, "sqrt", 0.80, "linear", 4.0);
        assert_eq!(market::market_price(0, i0, &o), 25, "base unchanged at I0");
        assert!(market::market_price(0, i0 + 100, &o) < market::market_price(0, i0 + 100, &t));
    }

    #[test]
    fn shed_access_tiles_are_the_four_centre_squares() {
        assert_eq!(shed_access_tiles(10), [(4, 4), (5, 4), (4, 5), (5, 5)]);
        assert_eq!(default_spawn(10), (4, 4));
    }
}
