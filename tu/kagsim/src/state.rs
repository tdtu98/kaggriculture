//! Game data tables and state, ported from `kaggle_environments/envs/kaggriculture/kaggriculture.py`.
//! Line references in comments point at that file.

pub const N_PRODUCTS: usize = 9;
pub const N_ITEMS: usize = 12; // 9 products + 3 animals (shed/inventory key space, `:157`)
pub const N_CROPS: usize = 5;
pub const N_ANIMALS: usize = 3;

// PRODUCTS order, `:25`. Item indices 0..8 are products; 9..11 are animals.
pub const ITEM_NAMES: [&str; N_ITEMS] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
    "GOOSE", "COW", "SHEEP",
];
pub const CROP_NAMES: [&str; N_CROPS] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
pub const ANIMAL_NAMES: [&str; N_ANIMALS] = ["GOOSE", "COW", "SHEEP"];

pub fn item_index(name: &str) -> Option<usize> {
    ITEM_NAMES.iter().position(|&n| n == name)
}
pub fn crop_index(name: &str) -> Option<usize> {
    CROP_NAMES.iter().position(|&n| n == name)
}
pub fn animal_index(name: &str) -> Option<usize> {
    ANIMAL_NAMES.iter().position(|&n| n == name)
}

/// CROPS, `:11`. Fields: seed cost, first_yield_day, max_yield_day, interval, max_yield, ongoing.
pub struct CropData {
    pub seed: i64,
    pub first_yield_day: i32,
    pub max_yield_day: i32,
    pub interval: i32,
    pub max_yield: i32,
    pub ongoing: bool,
}
pub const CROPS: [CropData; N_CROPS] = [
    CropData { seed: 10,  first_yield_day: 2,  max_yield_day: 4,  interval: 0, max_yield: 6, ongoing: false },
    CropData { seed: 20,  first_yield_day: 2,  max_yield_day: 3,  interval: 0, max_yield: 4, ongoing: false },
    CropData { seed: 50,  first_yield_day: 8,  max_yield_day: 8,  interval: 1, max_yield: 4, ongoing: true  },
    CropData { seed: 100, first_yield_day: 10, max_yield_day: 10, interval: 2, max_yield: 4, ongoing: true  },
    CropData { seed: 80,  first_yield_day: 10, max_yield_day: 12, interval: 0, max_yield: 6, ongoing: false },
];

/// ANIMALS, `:19`. `structure`: false = COOP, true = PASTURE. `product` is an item index.
pub struct AnimalData {
    pub cost: i64,
    pub pasture: bool,
    pub first_yield_day: i32,
    pub interval: i32,
    pub max_held: i32,
    pub product: usize,
}
pub const ANIMALS: [AnimalData; N_ANIMALS] = [
    AnimalData { cost: 300, pasture: false, first_yield_day: 4, interval: 1, max_held: 4, product: 5 }, // GOOSE -> EGG
    AnimalData { cost: 400, pasture: true,  first_yield_day: 8, interval: 2, max_held: 6, product: 6 }, // COW   -> MILK
    AnimalData { cost: 500, pasture: true,  first_yield_day: 6, interval: 3, max_held: 6, product: 7 }, // SHEEP -> WOOL
];

/// SHOPS, `:90`, in the alphabetical order that `sorted(remaining)` produces at `:867`.
/// 1.32.6: shops are drawn with replacement, so this caps total instances, not variety.
pub const MAX_SHOP_INSTANCES: usize = 8;

pub const SHOP_NAMES: [&str; 8] = [
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
];
/// Product indices demanded by each shop, matching SHOP_NAMES order.
pub const SHOP_DEMANDS: [&[usize]; 8] = [
    &[5, 0],        // BAKERY: EGG, WHEAT
    &[5, 0, 3],     // BRUNCH_SPOT: EGG, WHEAT, STRAWBERRY
    &[0, 1, 2, 3],  // FARMERS_MARKET: WHEAT, CARROT, TOMATO, STRAWBERRY
    &[3, 6, 0],     // ICE_CREAM_SHOP: STRAWBERRY, MILK, WHEAT
    &[1],           // PET_CAFE: CARROT (single-product -> 2x)
    &[6, 2, 0],     // PIZZA_SHOP: MILK, TOMATO, WHEAT
    &[3, 6],        // SMOOTHIE_SHOP: STRAWBERRY, MILK
    &[7],           // YARN_STORE: WOOL (single-product -> 2x)
];

pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];
/// LAND_ORDER = ["NE", "SW", "SE"], `:83`. Quadrant bit indices: 0=NW, 1=NE, 2=SW, 3=SE.
pub const LAND_ORDER: [u8; 3] = [1, 2, 3];
pub const QUADRANT_NAMES: [&str; 4] = ["NW", "NE", "SW", "SE"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileKind {
    Empty,
    Locked,
    Weed,
    Plant,
    Coop,
    Pasture,
}

/// One farm square. A single flat struct covers every case; `kind` tags which fields are live.
#[derive(Clone, Copy, Debug)]
pub struct Tile {
    pub kind: TileKind,
    pub crop: u8,               // valid iff kind == Plant
    pub animal: Option<u8>,     // Some(_) only after PLACE; empty structures stay None (`:440`)
    pub planted_day: i32,       // doubles as placed_day
    pub yield_units: i32,
    pub max_lifespan_step: i32, // -1 for ongoing crops (`:210`)
    pub fertilized_until_day: i32,
    pub consecutive_unwatered: i32, // doubles as consecutive_unfed
    pub pending_care_bonus: i32,
    pub watered_today: bool,        // doubles as fed_today
    pub cared_today: bool,
    pub fertilizer_available: bool,
}

impl Tile {
    pub const EMPTY: Tile = Tile {
        kind: TileKind::Empty,
        crop: 0,
        animal: None,
        planted_day: 0,
        yield_units: 0,
        max_lifespan_step: -1,
        fertilized_until_day: -1,
        consecutive_unwatered: 0,
        pending_care_bonus: 0,
        watered_today: false,
        cared_today: false,
        fertilizer_available: false,
    };

    pub const LOCKED: Tile = Tile { kind: TileKind::Locked, ..Tile::EMPTY };
    pub const WEED: Tile = Tile { kind: TileKind::Weed, ..Tile::EMPTY };

    /// `_new_plant`, `:201`. Note `consecutive_unwatered` starts at 1 — the planting day counts
    /// as already-missed — and one-time crops start with `yield_units = 1`.
    pub fn new_plant(crop: usize, day: i32, turns_per_day: i32) -> Tile {
        let cd = &CROPS[crop];
        Tile {
            kind: TileKind::Plant,
            crop: crop as u8,
            planted_day: day,
            watered_today: false,
            consecutive_unwatered: 1,
            yield_units: if cd.ongoing { 0 } else { 1 },
            max_lifespan_step: if cd.ongoing { -1 } else { (day + cd.max_yield_day + 1) * turns_per_day },
            fertilized_until_day: -1,
            ..Tile::EMPTY
        }
    }

    /// `_new_animal`, `:215`.
    pub fn new_animal(animal: usize, day: i32) -> Tile {
        Tile {
            kind: if ANIMALS[animal].pasture { TileKind::Pasture } else { TileKind::Coop },
            animal: Some(animal as u8),
            planted_day: day,
            yield_units: 0,
            consecutive_unwatered: 0,
            watered_today: false,
            cared_today: false,
            fertilizer_available: false,
            pending_care_bonus: 0,
            ..Tile::EMPTY
        }
    }

    pub fn is_occupied_structure(&self) -> bool {
        matches!(self.kind, TileKind::Coop | TileKind::Pasture) && self.animal.is_some()
    }
}

#[derive(Clone)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Tile>, // board_size * board_size, indexed [y * board_size + x]
    pub farmer: (i32, i32),
    pub hands: Vec<(i32, i32)>,
    pub unlocked: [bool; 4], // NW, NE, SW, SE
    pub hires_today: i32,
}

/// A farmer/hand inventory.
///
/// Deliberately an ordered vec, not an array: the reference is a Python dict, and
/// `_drop_inventories_to_shed` (`:825`) iterates `list(inv.items())` in **insertion order**. When
/// the shed is nearly full that order decides which items survive and which are discarded, so it
/// is observable state. `_inv_take` (`:289`) deletes a key at zero, and re-adding appends it at
/// the end — that reordering is replicated here.
pub type Inv = Vec<(u8, i64)>;

pub fn inv_get(inv: &Inv, item: usize) -> i64 {
    inv.iter().find(|(i, _)| *i as usize == item).map_or(0, |(_, n)| *n)
}

/// `_inv_add`, `:285`.
pub fn inv_add(inv: &mut Inv, item: usize, n: i64) {
    if let Some(e) = inv.iter_mut().find(|(i, _)| *i as usize == item) {
        e.1 += n;
    } else {
        inv.push((item as u8, n));
    }
}

/// `_inv_take`, `:289`. Removes the entry when it hits zero, matching `del inv[item]`.
pub fn inv_take(inv: &mut Inv, item: usize, n: i64) -> bool {
    let Some(pos) = inv.iter().position(|(i, _)| *i as usize == item) else { return false };
    if inv[pos].1 < n {
        return false;
    }
    inv[pos].1 -= n;
    if inv[pos].1 == 0 {
        inv.remove(pos);
    }
    true
}

#[derive(Clone)]
pub struct Private {
    /// The shed is a dict pre-populated with all 12 keys (`:157`) and never has keys deleted,
    /// so a fixed array is faithful here.
    pub shed: [i64; N_ITEMS],
    pub seeds: [i64; N_CROPS],
    pub inventories: Vec<Inv>, // [0] = main farmer, then hands
}

impl Private {
    pub fn shed_total(&self) -> i64 {
        self.shed.iter().sum()
    }
    /// `_farmer_inventory`, `:278` — grows the list on demand.
    pub fn inventory_mut(&mut self, idx: usize) -> &mut Inv {
        while self.inventories.len() <= idx {
            self.inventories.push(Inv::new());
        }
        &mut self.inventories[idx]
    }
}

#[derive(Clone)]
pub struct Market {
    pub inventory: [i64; N_PRODUCTS],
    pub prices: [i64; N_PRODUCTS],
}

#[derive(Clone)]
pub struct Config {
    pub episode_steps: i32,
    pub board_size: i32,
    pub starting_money: i64,
    pub max_market_orders: usize,
    pub turns_per_day: i32,
    pub shed_capacity: i64,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: i32,
    pub town_shop_sell_interval: i32,
    pub town_center_sell_interval: i32,
    pub farm_hand_cost_mult: i64,
    pub seed: i64,
    /// Resolved per-product price curves. `_resolve_market_params` (`:64`) merges sparse
    /// overrides onto the defaults, so this is always the full 9-entry table.
    pub market_params: Vec<crate::market::MarketParam>,
    /// True when `marketParams` was supplied and truthy. The reference only stores the table in
    /// `market["params"]` — and therefore only exposes it in the observation — in that case
    /// (`:169`, identity check against the module default).
    pub market_params_overridden: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3000,
            max_market_orders: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_shop_sell_interval: 4,
            town_center_sell_interval: 24,   // 1.32.6 default moved 12 -> 24
            farm_hand_cost_mult: 1,
            seed: 0,
            market_params: crate::market::default_market_params(),
            market_params_overridden: false,
        }
    }
}

/// Per-player diagnostics. Off by default: `collect_stats` gates the fingerprinting in
/// `rules::step`, so RL rollouts never pay for it.
#[derive(Clone, Default)]
pub struct Stats {
    pub actions_total: i64,
    pub actions_noop: i64,
    pub actions_move: i64,
    pub discarded_overflow: i64,
    pub sold_units: [i64; N_PRODUCTS],
    pub sold_revenue: [i64; N_PRODUCTS],
}

#[derive(Clone)]
pub struct GameState {
    pub cfg: Config,
    pub farms: Vec<Farm>,
    pub privates: Vec<Private>,
    pub market: Market,
    pub unlocked_shops: Vec<usize>,
    pub step: i32,
    pub day: i32,
    pub hour: i32,
    pub done: bool,
    pub stats: Vec<Stats>,
    pub collect_stats: bool,
    /// Set when a ported rule hits a case the reference raises on; surfaced by `Sim::step`.
    pub error: Option<String>,
}

/// `_quadrant_of`, `:113` -> bit index into `unlocked`.
pub fn quadrant_of(x: i32, y: i32, board_size: i32) -> usize {
    let half = board_size / 2;
    let north = y < half;
    let west = x < half;
    match (north, west) {
        (true, true) => 0,   // NW
        (true, false) => 1,  // NE
        (false, true) => 2,  // SW
        (false, false) => 3, // SE
    }
}

/// `_shed_access_tiles`, `:118` — the four inner corners, in NWSE order.
pub fn shed_access_tiles(board_size: i32) -> [(i32, i32); 4] {
    let h = board_size / 2;
    [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
}

pub fn is_shed_adjacent(pos: (i32, i32), board_size: i32) -> bool {
    shed_access_tiles(board_size).contains(&pos)
}

/// `_default_spawn`, `:147` — first shed-access tile inside the NW quadrant.
pub fn default_spawn(board_size: i32) -> (i32, i32) {
    for t in shed_access_tiles(board_size) {
        if quadrant_of(t.0, t.1, board_size) == 0 {
            return t;
        }
    }
    (0, 0)
}

/// `_fib`, `:667` — indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5.
pub fn fib(n: i32) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        let t = a.wrapping_add(b);
        a = b;
        b = t;
    }
    a
}

impl GameState {
    pub fn new(cfg: Config) -> GameState {
        let bs = cfg.board_size;
        let tiles: Vec<Tile> = (0..bs * bs)
            .map(|i| {
                let (x, y) = (i % bs, i / bs);
                if quadrant_of(x, y, bs) == 0 { Tile::EMPTY } else { Tile::LOCKED }
            })
            .collect();
        let farm = Farm {
            money: cfg.starting_money as f64,
            tiles,
            farmer: default_spawn(bs),
            hands: Vec::new(),
            unlocked: [true, false, false, false],
            hires_today: 0,
        };
        let private = Private {
            shed: [0; N_ITEMS],
            seeds: [0; N_CROPS],
            inventories: vec![Inv::new()],
        };
        GameState {
            farms: vec![farm.clone(), farm],
            privates: vec![private.clone(), private],
            market: Market {
                // `_new_market` (`:164`) seeds inventory from each product's own I0 and prices
                // from its base — not from market_price(), though they agree at I0.
                inventory: std::array::from_fn(|i| cfg.market_params[i].i0),
                prices: std::array::from_fn(|i| cfg.market_params[i].base as i64),
            },
            unlocked_shops: Vec::new(),
            step: 0,
            day: 0,
            hour: 0,
            done: false,
            stats: vec![Stats::default(), Stats::default()],
            collect_stats: false,
            error: None,
            cfg,
        }
    }

    #[inline]
    pub fn tile(&self, p: usize, x: i32, y: i32) -> Tile {
        self.farms[p].tiles[(y * self.cfg.board_size + x) as usize]
    }
    #[inline]
    pub fn set_tile(&mut self, p: usize, x: i32, y: i32, t: Tile) {
        let bs = self.cfg.board_size;
        self.farms[p].tiles[(y * bs + x) as usize] = t;
    }
}
