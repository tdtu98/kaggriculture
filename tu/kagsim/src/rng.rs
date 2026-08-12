//! CPython-compatible Mersenne Twister.
//!
//! Required for bit-exact parity (see `docs/decisions.md` D5). `kaggriculture.py:848` builds
//! `random.Random((seed * 1_000_003) ^ day)` and then `_spawn_weeds` (`:817`) draws `random()`
//! **once per empty unlocked tile**. Which tiles are empty depends on gameplay, so the draw
//! sequence cannot be precomputed from `(seed, day)` — we have to reproduce CPython's generator.
//!
//! Ported from CPython `Modules/_randommodule.c` and `Lib/random.py`.

use pyo3::prelude::*;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MtRandom {
    mt: [u32; N],
    index: usize,
}

impl MtRandom {
    /// Mirrors `random_seed()` for integer arguments: take `abs(n)`, decompose into 32-bit
    /// little-endian words, and run `init_by_array`. A zero seed uses the single-word key `[0]`.
    pub fn new(seed: i128) -> Self {
        let mut n = seed.unsigned_abs();
        let mut key: Vec<u32> = Vec::new();
        while n > 0 {
            key.push((n & 0xffff_ffff) as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = MtRandom { mt: [0; N], index: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = UPPER_MASK;
        self.index = N;
    }

    fn generate(&mut self) {
        for i in 0..N {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
            let mut next = self.mt[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.mt[i] = next;
        }
        self.index = 0;
    }

    pub fn genrand_uint32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython `random_random`: two draws combined into a 53-bit double.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64;
        let b = (self.genrand_uint32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// CPython `getrandbits` for `k <= 32`, which is all this simulator needs
    /// (`choice` is only ever called on the shop list, length <= 8).
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        assert!(k <= 32, "getrandbits only implemented for k <= 32");
        self.genrand_uint32() >> (32 - k)
    }

    /// `Lib/random.py::_randbelow_with_getrandbits` — rejection sampling, no modulo bias.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros(); // n.bit_length()
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.choice(seq)` returns `seq[randbelow(len(seq))]`; we return the index.
    pub fn choice_index(&mut self, len: u32) -> u32 {
        self.randbelow(len)
    }
}

/// Python-facing handle, used by the parity tests to compare against `random.Random`.
#[pyclass(name = "PyRandom")]
pub struct PyRandom {
    inner: MtRandom,
}

#[pymethods]
impl PyRandom {
    #[new]
    fn new(seed: i128) -> Self {
        PyRandom { inner: MtRandom::new(seed) }
    }
    fn random(&mut self) -> f64 {
        self.inner.random()
    }
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.inner.getrandbits(k)
    }
    fn randbelow(&mut self, n: u32) -> u32 {
        self.inner.randbelow(n)
    }
    fn choice_index(&mut self, len: u32) -> u32 {
        self.inner.choice_index(len)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRandom>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_reference_vector() {
        // Standard MT19937 init_by_array test vector (init_key = [0x123, 0x234, 0x345, 0x456]).
        let mut r = MtRandom { mt: [0; N], index: N };
        r.init_by_array(&[0x123, 0x234, 0x345, 0x456]);
        assert_eq!(r.genrand_uint32(), 1067595299);
        assert_eq!(r.genrand_uint32(), 955945823);
        assert_eq!(r.genrand_uint32(), 477289528);
    }

    #[test]
    fn zero_seed_uses_single_word_key() {
        let mut a = MtRandom::new(0);
        let mut b = MtRandom::new(0);
        assert_eq!(a.random(), b.random());
    }
}
