//! Market pricing and order processing. Ports `kaggriculture.py:38-232` and `:521-664`.

use crate::round_half_even;
use crate::state::*;

pub const DEFAULT_I0: i64 = 10_000;
pub const PRICE_FLOOR: i64 = 1;

/// `_shape`, `:54`. Unknown names fall through to identity, matching the reference's `return x`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shape {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Identity,
}

pub fn shape_from_name(name: &str) -> Shape {
    match name {
        "linear" => Shape::Linear,
        "sq" => Shape::Sq,
        "sqrt" => Shape::Sqrt,
        "log" => Shape::Log,
        "log10" => Shape::Log10,
        _ => Shape::Identity,
    }
}

fn shape(f: Shape, x: f64) -> f64 {
    let x = x.max(0.0);
    match f {
        Shape::Linear | Shape::Identity => x,
        Shape::Sq => x * x,
        Shape::Sqrt => x.sqrt(),
        Shape::Log => (1.0 + x).ln(),
        Shape::Log10 => (1.0 + x).log10(),
    }
}

/// One product's price curve. Holds both the parsed shape (for the hot path) and the raw name,
/// because the reference exposes the resolved table inside `obs["market"]["params"]`, so the
/// original string is observable state.
#[derive(Clone, Debug)]
pub struct MarketParam {
    pub base: f64,
    pub i0: i64,
    pub t: f64,
    pub below_func: String,
    pub below_target: f64,
    pub above_func: String,
    pub above_target: f64,
    below_shape: Shape,
    above_shape: Shape,
}

impl MarketParam {
    pub fn new(base: f64, i0: i64, t: f64, bf: &str, bt: f64, af: &str, at: f64) -> Self {
        MarketParam {
            base,
            i0,
            t,
            below_func: bf.to_string(),
            below_target: bt,
            above_func: af.to_string(),
            above_target: at,
            below_shape: shape_from_name(bf),
            above_shape: shape_from_name(af),
        }
    }
    /// Recompute the cached shapes after a sparse patch changes a func name.
    pub fn resync(&mut self) {
        self.below_shape = shape_from_name(&self.below_func);
        self.above_shape = shape_from_name(&self.above_func);
    }
}

/// MARKET_PARAMS, `:41`, in PRODUCTS order.
pub fn default_market_params() -> Vec<MarketParam> {
    vec![
        MarketParam::new(25.0, DEFAULT_I0, 400.0, "sqrt", 0.80, "log", 0.20),      // WHEAT
        MarketParam::new(35.0, DEFAULT_I0, 450.0, "log", 0.20, "sqrt", 0.70),      // CARROT
        MarketParam::new(60.0, DEFAULT_I0, 200.0, "linear", 0.40, "sqrt", 0.60),   // TOMATO
        MarketParam::new(120.0, DEFAULT_I0, 100.0, "sqrt", 0.70, "linear", 1.60),  // STRAWBERRY
        MarketParam::new(250.0, DEFAULT_I0, 300.0, "log", 0.20, "sq", 3.60),       // MELON
        MarketParam::new(50.0, DEFAULT_I0, 332.0, "linear", 0.40, "log", 0.20),    // EGG
        MarketParam::new(160.0, DEFAULT_I0, 122.0, "sqrt", 0.60, "linear", 1.60),  // MILK
        MarketParam::new(200.0, DEFAULT_I0, 105.0, "log", 0.20, "sq", 3.20),       // WOOL
        MarketParam::new(100.0, DEFAULT_I0, 200.0, "linear", 0.40, "linear", 0.40), // FERTILIZER
    ]
}

/// `market_price`, `:178`.
///
/// The `int(round(price))` at `:192` is Python banker's rounding — see `round_half_even`.
pub fn market_price(item: usize, inventory: i64, params: &[MarketParam]) -> i64 {
    let p = &params[item];
    let price = if inventory < p.i0 {
        let amp = p.below_target * p.base / shape(p.below_shape, p.t);
        p.base + amp * shape(p.below_shape, (p.i0 - inventory) as f64)
    } else {
        let amp = p.above_target * p.base / shape(p.above_shape, p.t);
        p.base - amp * shape(p.above_shape, (inventory - p.i0) as f64)
    };
    (round_half_even(price) as i64).max(PRICE_FLOOR)
}

pub fn refresh_prices(m: &mut Market, params: &[MarketParam]) {
    for i in 0..N_PRODUCTS {
        m.prices[i] = market_price(i, m.inventory[i], params);
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OrderType {
    Hire,
    BuyLand,
    BuySeed,
    BuyProduct,
    BuyAnimal,
    Sell,
}

#[derive(Clone, Debug)]
pub struct Order {
    pub kind: OrderType,
    /// Raw item name; validity is checked at quote time, not parse time (`:573`).
    pub item: String,
    pub remaining: i64,
}

/// `_parse_order`, `:608`. Returns None for anything malformed.
pub fn parse_order(op: &str, item: Option<&str>, n: Option<i64>) -> Option<Order> {
    match op {
        "HIRE" => Some(Order { kind: OrderType::Hire, item: String::new(), remaining: 0 }),
        "BUY_LAND" => Some(Order { kind: OrderType::BuyLand, item: String::new(), remaining: 0 }),
        "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "SELL" => {
            let (item, n) = (item?, n?);
            if n <= 0 {
                return None;
            }
            let kind = match op {
                "BUY_SEED" => OrderType::BuySeed,
                "BUY_PRODUCT" => OrderType::BuyProduct,
                "BUY_ANIMAL" => OrderType::BuyAnimal,
                _ => OrderType::Sell,
            };
            Some(Order { kind, item: item.to_string(), remaining: n })
        }
        _ => None,
    }
}

/// `_do_hire`, `:679`.
fn do_hire(st: &mut GameState, p: usize) {
    let cost = st.cfg.farm_hand_cost_mult * fib(st.farms[p].hires_today);
    if st.farms[p].money < cost as f64 {
        return;
    }
    st.farms[p].money -= cost as f64;
    st.farms[p].hires_today += 1;
    let pos = spawn_hand(&st.farms[p], st.cfg.board_size);
    st.farms[p].hands.push(pos);
    st.privates[p].inventories.push(Inv::new());
}

/// `_spawn_hand`, `:510` — least-occupied shed-access tile, ties broken by NWSE order.
/// Deliberately ignores whether the tile is locked (`:159`).
fn spawn_hand(farm: &Farm, board_size: i32) -> (i32, i32) {
    let tiles = shed_access_tiles(board_size);
    let mut occ = [0i32; 4];
    let mut all = vec![farm.farmer];
    all.extend(farm.hands.iter().copied());
    for pos in all {
        if let Some(i) = tiles.iter().position(|&t| t == pos) {
            occ[i] += 1;
        }
    }
    let mut best = 0usize;
    for i in 1..4 {
        if occ[i] < occ[best] {
            best = i;
        }
    }
    tiles[best]
}

/// `_do_buy_land`, `:689`.
fn do_buy_land(st: &mut GameState, p: usize) {
    let n_extra = st.farms[p].unlocked.iter().filter(|&&u| u).count() - 1;
    if n_extra >= LAND_ORDER.len() {
        return;
    }
    let cost = LAND_PRICES[n_extra];
    if st.farms[p].money < cost as f64 {
        return;
    }
    st.farms[p].money -= cost as f64;
    let q = LAND_ORDER[n_extra] as usize;
    st.farms[p].unlocked[q] = true;
    let bs = st.cfg.board_size;
    for y in 0..bs {
        for x in 0..bs {
            let idx = (y * bs + x) as usize;
            if quadrant_of(x, y, bs) == q && st.farms[p].tiles[idx].kind == TileKind::Locked {
                st.farms[p].tiles[idx] = Tile::EMPTY;
            }
        }
    }
}

/// `_commit_unit`, `:629`. Returns false when the order cannot continue.
fn commit_unit(st: &mut GameState, p: usize, kind: OrderType, item: usize, price: i64) -> bool {
    let cap = st.cfg.shed_capacity;
    match kind {
        OrderType::Sell => {
            if st.privates[p].shed[item] <= 0 {
                return false;
            }
            st.privates[p].shed[item] -= 1;
            st.farms[p].money += price as f64;
            if st.collect_stats {
                st.stats[p].sold_units[item] += 1;
                st.stats[p].sold_revenue[item] += price;
            }
            // Sales at the floor do not add supply (`:636`).
            if price > 1 {
                st.market.inventory[item] += 1;
            }
            true
        }
        OrderType::BuyProduct => {
            if st.farms[p].money < price as f64 {
                return false;
            }
            if st.privates[p].shed_total() >= cap {
                return false;
            }
            st.farms[p].money -= price as f64;
            st.privates[p].shed[item] += 1;
            st.market.inventory[item] -= 1;
            true
        }
        OrderType::BuySeed => {
            if st.farms[p].money < price as f64 {
                return false;
            }
            st.farms[p].money -= price as f64;
            st.privates[p].seeds[item] += 1;
            true
        }
        OrderType::BuyAnimal => {
            if st.farms[p].money < price as f64 {
                return false;
            }
            if st.privates[p].shed_total() >= cap {
                return false;
            }
            st.farms[p].money -= price as f64;
            st.privates[p].shed[item] += 1;
            true
        }
        _ => false,
    }
}

/// `_process_market`, `:521`.
///
/// Order slots are processed in lockstep: for slot `i`, HIRE/BUY_LAND resolve atomically in player
/// order, then a per-unit loop quotes **both** players against the same pre-commit inventory and
/// commits both before re-quoting. Prices refresh once per slot, not per unit.
pub fn process_market(st: &mut GameState, queues: &[Vec<Option<Order>>]) {
    let params = st.cfg.market_params.clone();
    let max_orders = st.cfg.max_market_orders.max(1);
    let queues: Vec<Vec<Option<Order>>> = queues
        .iter()
        .map(|q| q.iter().take(max_orders).cloned().collect())
        .collect();

    let max_len = queues.iter().map(|q| q.len()).max().unwrap_or(0);

    for i in 0..max_len {
        // Snapshot this slot's order for each player; None means "no order / aborted".
        let mut active: Vec<Option<Order>> = queues
            .iter()
            .map(|q| q.get(i).cloned().flatten())
            .collect();

        for p in 0..active.len() {
            if let Some(o) = &active[p] {
                match o.kind {
                    OrderType::Hire => {
                        do_hire(st, p);
                        active[p] = None;
                    }
                    OrderType::BuyLand => {
                        do_buy_land(st, p);
                        active[p] = None;
                    }
                    _ => {}
                }
            }
        }

        let mut guard = 0;
        loop {
            guard += 1;
            if guard >= 100_000 {
                break; // matches the reference's runaway guard (`:564`)
            }

            // Quote phase — every player sees the same pre-commit inventory.
            let mut quoted: Vec<Option<(OrderType, usize, i64)>> = vec![None; active.len()];
            for p in 0..active.len() {
                let Some(o) = &active[p] else { continue };
                if o.remaining <= 0 {
                    continue;
                }
                let q = match o.kind {
                    OrderType::Sell => item_index(&o.item)
                        .filter(|&it| it < N_PRODUCTS)
                        .map(|it| (OrderType::Sell, it, market_price(it, st.market.inventory[it], &params))),
                    OrderType::BuyProduct => match o.item.as_str() {
                        // Only WHEAT and FERTILIZER are buyable (`:575`); quoted at post-buy
                        // inventory so a buy/sell round trip nets zero (`:578`).
                        "WHEAT" => Some((OrderType::BuyProduct, 0, market_price(0, st.market.inventory[0] - 1, &params))),
                        "FERTILIZER" => Some((OrderType::BuyProduct, 8, market_price(8, st.market.inventory[8] - 1, &params))),
                        _ => None,
                    },
                    OrderType::BuySeed => crop_index(&o.item)
                        .map(|c| (OrderType::BuySeed, c, CROPS[c].seed)),
                    // The shed is keyed by item index; animals live at 9..11, so the animal
                    // index must be mapped through `item_index`, not used directly.
                    OrderType::BuyAnimal => animal_index(&o.item)
                        .map(|a| (OrderType::BuyAnimal, item_index(&o.item).unwrap(), ANIMALS[a].cost)),
                    _ => None,
                };
                match q {
                    Some(v) => quoted[p] = Some(v),
                    None => active[p] = None, // malformed sub-op aborts the order (`:584`)
                }
            }

            if quoted.iter().all(|q| q.is_none()) {
                break;
            }

            // Commit phase.
            let mut committed_any = false;
            for p in 0..quoted.len() {
                let Some((kind, item, price)) = quoted[p] else { continue };
                if commit_unit(st, p, kind, item, price) {
                    if let Some(o) = &mut active[p] {
                        o.remaining -= 1;
                    }
                    committed_any = true;
                } else {
                    active[p] = None;
                }
            }
            if !committed_any {
                break;
            }
        }

        refresh_prices(&mut st.market, &params);
    }
}

/// `_town_consume`, `:705`.
pub fn town_consume(st: &mut GameState, step: i32) {
    let shop_interval = st.cfg.town_shop_sell_interval.max(1);
    let center_interval = st.cfg.town_center_sell_interval.max(1);

    if step % shop_interval == 0 {
        // `unlocked_shops` may list the same shop more than once since 1.32.6 (shops are drawn
        // with replacement); each instance consumes independently, which this loop already does.
        for &s in &st.unlocked_shops {
            let products = SHOP_DEMANDS[s];
            let mult = if products.len() == 1 { 2 } else { 1 };
            for &item in products {
                st.market.inventory[item] -= mult;
            }
        }
    }

    if step % center_interval == 0 {
        // 1.32.6 removed TOWN_CENTER_DEMAND_SCHEDULE: the centre now buys exactly one of each
        // product per tick for the whole season, and the default interval moved 12 -> 24. Combined
        // that is ~140 -> 30 units of seasonal demand per product, a 4.7x cut.
        // TOWN_CENTER_PRODUCTS excludes FERTILIZER (`:100`).
        for item in 0..N_PRODUCTS - 1 {
            st.market.inventory[item] -= 1;
        }
    }

    let params = st.cfg.market_params.clone();
    refresh_prices(&mut st.market, &params);
}
