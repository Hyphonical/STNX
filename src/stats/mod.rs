//! Statistical analysis and ECDF table construction for Constellation Encoding.
//!
//! Implements the order-statistic lookup table builder, two-sample K–S test,
//! chi-squared byte-frequency test, and padding CSPRNG as specified in
//! PLAN.md Sections 4.2 and 10.2.

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// ECDF Order-Statistic Lookup Table  (PLAN.md §4.2.2)
// ---------------------------------------------------------------------------

/// Build a 256-entry ECDF order-statistic lookup table from donor scalars.
///
/// Each entry is a value drawn directly from the sorted donor population at
/// quantile midpoints.  The uniqueness guarantee ensures all 256 entries are
/// distinct, enabling an exact inverse lookup during decoding.
///
/// Returns `None` if the donor has fewer than 256 distinct values (ineligible).
///
/// # Panics
///
/// Panics if `donor` is empty (dividing by zero in the index formula).
pub fn build_ecdf_table(donor: &[f32]) -> Option<[f32; 256]> {
	assert!(!donor.is_empty(), "donor tensor must not be empty");

	let n = donor.len();
	let mut sorted = donor.to_vec();
	sorted.sort_unstable_by(|a, b| a.total_cmp(b));

	let mut table = [0.0f32; 256];
	let mut search_start = 0usize;
	let mut prev_val = f32::NEG_INFINITY;

	for (k, entry) in table.iter_mut().enumerate() {
		// Target index at quantile midpoint  (PLAN.md §4.2.2)
		let target = ((k as f64 + 0.5) * n as f64 / 256.0) as usize;
		let mut idx = target.max(search_start);

		// Advance past any value equal to the previous entry (uniqueness)
		while idx < n && sorted[idx] == prev_val {
			idx += 1;
		}

		if idx >= n {
			// Fewer than 256 distinct values → ineligible donor
			return None;
		}

		*entry = sorted[idx];
		prev_val = sorted[idx];
		search_start = idx + 1;
	}

	Some(table)
}

/// Build the 256-entry decoding lookup table.
///
/// This is the same table as `build_ecdf_table`, but pre-sorted so that
/// decoding can use binary search.  An alternative to re-sorting every time.
pub fn build_sorted_ecdf_table(donor: &[f32]) -> Option<([f32; 256], [u8; 256])> {
	let table = build_ecdf_table(donor)?;

	// Build index pairs and sort by value.
	let mut pairs: Vec<(f32, u8)> = table.iter().copied().zip(0u8..=255).collect();
	pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

	let mut sorted_vals = [0.0f32; 256];
	let mut sorted_idx = [0u8; 256];
	for (i, (val, byte)) in pairs.iter().enumerate() {
		sorted_vals[i] = *val;
		sorted_idx[i] = *byte;
	}

	Some((sorted_vals, sorted_idx))
}

// ---------------------------------------------------------------------------
// Encoding / Decoding  (PLAN.md §4.2.2 – §4.2.3)
// ---------------------------------------------------------------------------

/// Encode a slice of payload bytes using the ECDF table.
///
/// Each byte `b` maps to `table[b]`.
pub fn encode_chunk(bytes: &[u8], table: &[f32; 256]) -> Vec<f32> {
	bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Decode a slice of stego floats back to raw bytes via exact lookup.
///
/// The `sorted_vals` array must contain the 256 distinct table values in
/// ascending order, and `sorted_idx` maps each position back to the
/// original byte value.
///
/// # Panics
///
/// Panics if any float is not found in the table (should never happen with
/// valid stego data).
pub fn decode_chunk(stego: &[f32], sorted_vals: &[f32; 256], sorted_idx: &[u8; 256]) -> Vec<u8> {
	stego
		.iter()
		.map(|&val| {
			// Binary search on the 256-entry sorted table.
			let pos = sorted_vals
				.binary_search_by(|probe| probe.total_cmp(&val))
				.expect("stego float must be an exact table entry");
			sorted_idx[pos]
		})
		.collect()
}

// ---------------------------------------------------------------------------
// Padding CSPRNG  (PLAN.md §4.2.4)
// ---------------------------------------------------------------------------

/// A simple CSPRNG based on SHA-256 in counter mode, seeded by a 32-byte key.
///
/// Generates an infinite stream of pseudorandom bytes deterministically from
/// the seed.  Used for synthetic padding in the final chunk.
pub struct PaddingRng {
	key: [u8; 32],
	counter: u64,
	buffer: [u8; 32],
	pos: usize,
}

impl PaddingRng {
	/// Create a new CSPRNG from a 32-byte seed.
	pub fn new(key: &[u8; 32]) -> Self {
		let mut rng = Self {
			key: *key,
			counter: 0,
			buffer: [0u8; 32],
			pos: 32, // force a refill on first call
		};
		rng.refill();
		rng
	}

	fn refill(&mut self) {
		let mut hasher = Sha256::new();
		hasher.update(self.key);
		hasher.update(self.counter.to_le_bytes());
		self.buffer = hasher.finalize().into();
		self.counter += 1;
		self.pos = 0;
	}

	/// Generate `len` pseudorandom bytes.
	pub fn generate(&mut self, len: usize) -> Vec<u8> {
		let mut out = Vec::with_capacity(len);
		for _ in 0..len {
			if self.pos >= 32 {
				self.refill();
			}
			out.push(self.buffer[self.pos]);
			self.pos += 1;
		}
		out
	}

	/// Generate a single pseudorandom byte.
	pub fn next_u8(&mut self) -> u8 {
		if self.pos >= 32 {
			self.refill();
		}
		let b = self.buffer[self.pos];
		self.pos += 1;
		b
	}
}

// ---------------------------------------------------------------------------
// Two-Sample Kolmogorov–Smirnov Test  (PLAN.md §10.2)
// ---------------------------------------------------------------------------

/// Compute the two-sample Kolmogorov–Smirnov D statistic.
///
/// The statistic is:
///
/// $$D = \max_x |F_1(x) - F_2(x)|$$
///
/// where $F_1, F_2$ are the empirical CDFs of the two samples.
///
/// The inputs are the raw (unsorted) samples.  They will be sorted internally.
pub fn ks_statistic(sample_a: &[f64], sample_b: &[f64]) -> f64 {
	let mut a = sample_a.to_vec();
	let mut b = sample_b.to_vec();
	a.sort_unstable_by(|x, y| x.total_cmp(y));
	b.sort_unstable_by(|x, y| x.total_cmp(y));

	let n = a.len() as f64;
	let m = b.len() as f64;

	let mut i = 0usize;
	let mut j = 0usize;
	let mut d_max = 0.0f64;

	while i < a.len() && j < b.len() {
		if a[i] < b[j] {
			i += 1;
		} else if a[i] > b[j] {
			j += 1;
		} else {
			// Equal values: advance both pointers simultaneously
			i += 1;
			j += 1;
		}

		let d = ((i as f64 / n) - (j as f64 / m)).abs();
		if d > d_max {
			d_max = d;
		}
	}

	// Drain remaining elements
	while i < a.len() {
		i += 1;
		let d = ((i as f64 / n) - 1.0).abs();
		if d > d_max {
			d_max = d;
		}
	}
	while j < b.len() {
		j += 1;
		let d = (1.0 - (j as f64 / m)).abs();
		if d > d_max {
			d_max = d;
		}
	}

	d_max
}

/// Critical value for the two-sample K–S test at α = 0.05.
///
/// Uses the approximation:
///
/// $$D_{\text{crit}} = 1.36 \cdot \sqrt{\frac{n + m}{n \cdot m}}$$
///
/// where `n` and `m` are the sample sizes.
pub fn ks_critical_value_005(n: usize, m: usize) -> f64 {
	1.36 * ((n + m) as f64 / (n as f64 * m as f64)).sqrt()
}

/// Run the two-sample K–S test at α = 0.05.
///
/// Returns `true` if the null hypothesis (same distribution) is **not**
/// rejected — i.e.\ the samples are statistically indistinguishable at
/// the 5% significance level.
pub fn ks_test_passes(sample_a: &[f64], sample_b: &[f64]) -> bool {
	let d = ks_statistic(sample_a, sample_b);
	let crit = ks_critical_value_005(sample_a.len(), sample_b.len());
	d <= crit
}

// ---------------------------------------------------------------------------
// Chi-Squared Byte-Frequency Test  (PLAN.md §10.2)
// ---------------------------------------------------------------------------

/// Compute the chi-squared statistic on byte frequencies.
///
/// $$ \chi^2 = \sum_{b=0}^{255} \frac{(\text{obs}_b - \text{exp}_b)^2}{\text{exp}_b} $$
///
/// where `obs` are the observed byte frequencies from the stego tensor's
/// `raw_data` and `exp` are the expected frequencies from a genuine weight
/// tensor simulation.
///
/// # Panics
///
/// Panics if `obs` and `exp` have different lengths or if any expected
/// frequency is zero.
pub fn chi_squared_byte_test(obs: &[u64], exp: &[u64]) -> f64 {
	assert_eq!(
		obs.len(),
		exp.len(),
		"observed and expected frequency arrays must have the same length"
	);
	assert_eq!(
		obs.len(),
		256,
		"byte frequency arrays must have 256 entries"
	);

	let mut statistic = 0.0f64;
	for i in 0..256 {
		let o = obs[i] as f64;
		let e = exp[i] as f64;
		if e <= 0.0 {
			// If expected frequency is zero, the observed must also be zero —
			// otherwise the test fails trivially.
			if o > 0.0 {
				return f64::INFINITY;
			}
			continue;
		}
		statistic += (o - e).powi(2) / e;
	}
	statistic
}

/// Count byte frequencies from a raw byte slice.
///
/// Returns a 256-element array where `result[b]` is the count of byte `b`.
pub fn byte_frequencies(data: &[u8]) -> [u64; 256] {
	let mut counts = [0u64; 256];
	for &b in data {
		counts[b as usize] += 1;
	}
	counts
}

/// Critical value for the chi-squared test with 255 degrees of freedom at α = 0.05.
///
/// Approximation for large df:  χ²_crit ≈ df + z * sqrt(2 * df)
/// where z = 1.96 for α = 0.05 (two-tailed) / upper-tail 0.05.
///
/// For df=255: ~ 293.25.  We use a precise pre-computed value.
pub const CHI_SQUARED_CRITICAL_005: f64 = 293.247_835;

/// Run the chi-squared byte-frequency test at α = 0.05.
///
/// Returns `true` if the null hypothesis (same byte distribution) is **not**
/// rejected.
pub fn chi_squared_test_passes(obs: &[u64], exp: &[u64]) -> bool {
	let statistic = chi_squared_byte_test(obs, exp);
	statistic <= CHI_SQUARED_CRITICAL_005
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	// ── ECDF table ─────────────────────────────────────────────────────────

	#[test]
	fn test_build_ecdf_table_256_distinct_values() {
		// Exactly 256 distinct values: 0.0 .. 255.0
		let donor: Vec<f32> = (0..256).map(|i| i as f32).collect();
		let table = build_ecdf_table(&donor).expect("should build table");
		assert_eq!(table.len(), 256, "table must have 256 entries");

		// All entries should be distinct
		let mut sorted = table.to_vec();
		sorted.sort_unstable_by(|a, b| a.total_cmp(b));
		sorted.dedup();
		assert_eq!(sorted.len(), 256, "all 256 table entries must be distinct");
	}

	#[test]
	fn test_build_ecdf_table_fewer_than_256_values() {
		let donor: Vec<f32> = (0..255).map(|i| i as f32).collect();
		assert!(
			build_ecdf_table(&donor).is_none(),
			"fewer than 256 distinct values must return None"
		);
	}

	#[test]
	fn test_build_ecdf_table_large_normal_distribution() {
		// Simulate a realistic weight tensor with ~10000 values from N(0,1)
		let mut donor: Vec<f32> = Vec::with_capacity(10_000);
		for i in 0..10_000 {
			// Box-Muller-ish spread, but just use evenly spaced values
			donor.push((i as f32 / 1000.0) - 5.0);
		}
		let table = build_ecdf_table(&donor).expect("large donor should build");
		assert_eq!(table.len(), 256);

		// Verify monotonicity (sorted order means non-decreasing)
		for i in 1..256 {
			assert!(
				table[i - 1] < table[i],
				"table must be strictly increasing (unique entries): table[{}]={} >= table[{}]={}",
				i - 1,
				table[i - 1],
				i,
				table[i]
			);
		}
	}

	#[test]
	fn test_build_ecdf_table_duplicate_donor_values() {
		// Donor with many duplicates but still ≥ 256 distinct
		let mut donor = Vec::new();
		for i in 0..300u32 {
			// Repeat each value 10 times
			for _ in 0..10 {
				donor.push(i as f32);
			}
		}
		let table = build_ecdf_table(&donor).expect("should build from 300 distinct");
		assert_eq!(table.len(), 256);

		// Verify strict monotonicity
		for i in 1..256 {
			assert!(table[i - 1] < table[i], "table must be strictly increasing");
		}
	}

	// ── Sorted table ───────────────────────────────────────────────────────

	#[test]
	fn test_build_sorted_ecdf_table() {
		let donor: Vec<f32> = (0..256).map(|i| i as f32).collect();
		let (sorted_vals, sorted_idx) = build_sorted_ecdf_table(&donor).unwrap();

		assert_eq!(sorted_vals.len(), 256);
		assert_eq!(sorted_idx.len(), 256);

		// sorted_vals must be ascending
		for i in 1..256 {
			assert!(sorted_vals[i - 1] < sorted_vals[i]);
		}

		// Verify that sorted_idx maps back: for each original byte b,
		// sorted_idx at the position of table[b] should equal b.
		let table = build_ecdf_table(&donor).unwrap();
		for b in 0u8..=255 {
			let val = table[b as usize];
			let pos = sorted_vals.binary_search_by(|p| p.total_cmp(&val)).unwrap();
			assert_eq!(sorted_idx[pos], b, "sorted_idx must map back to byte {b}");
		}
	}

	// ── Encode / Decode round-trip ─────────────────────────────────────────

	#[test]
	fn test_encode_decode_roundtrip() {
		let donor: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
		let table = build_ecdf_table(&donor).unwrap();
		let (sorted_vals, sorted_idx) = build_sorted_ecdf_table(&donor).unwrap();

		// Encode a payload
		let payload: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
		let encoded = encode_chunk(&payload, &table);
		assert_eq!(encoded.len(), payload.len());

		// Decode
		let decoded = decode_chunk(&encoded, &sorted_vals, &sorted_idx);
		assert_eq!(decoded, payload, "encode → decode must roundtrip exactly");
	}

	#[test]
	fn test_encode_decode_all_256_bytes() {
		let donor: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
		let table = build_ecdf_table(&donor).unwrap();
		let (sorted_vals, sorted_idx) = build_sorted_ecdf_table(&donor).unwrap();

		let payload: Vec<u8> = (0u8..=255).collect();
		let encoded = encode_chunk(&payload, &table);
		let decoded = decode_chunk(&encoded, &sorted_vals, &sorted_idx);
		assert_eq!(decoded, payload, "all 256 bytes must roundtrip");
	}

	// ── Padding CSPRNG ─────────────────────────────────────────────────────

	#[test]
	fn test_padding_rng_deterministic() {
		let key = [0xABu8; 32];
		let mut rng1 = PaddingRng::new(&key);
		let mut rng2 = PaddingRng::new(&key);

		let a = rng1.generate(100);
		let b = rng2.generate(100);
		assert_eq!(a, b, "same seed must produce same output");
	}

	#[test]
	fn test_padding_rng_different_seeds() {
		let key1 = [0xABu8; 32];
		let key2 = [0xCDu8; 32];
		let mut rng1 = PaddingRng::new(&key1);
		let mut rng2 = PaddingRng::new(&key2);

		let a = rng1.generate(100);
		let b = rng2.generate(100);
		assert_ne!(a, b, "different seeds must produce different output");
	}

	#[test]
	fn test_padding_rng_large_output() {
		let key = [0x42u8; 32];
		let mut rng = PaddingRng::new(&key);
		let data = rng.generate(10_000);
		assert_eq!(data.len(), 10_000);

		// Should have reasonable entropy (not all zeros)
		let distinct: std::collections::HashSet<u8> = data.iter().copied().collect();
		assert!(distinct.len() > 200, "should produce diverse bytes");
	}

	// ── K-S test ────────────────────────────────────────────────────────────

	#[test]
	fn test_ks_statistic_identical() {
		let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
		let b: Vec<f64> = (0..1000).map(|i| i as f64).collect();
		let d = ks_statistic(&a, &b);
		assert!(d < 1e-10, "identical samples must have D ≈ 0, got {d}");
	}

	#[test]
	fn test_ks_statistic_different() {
		let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
		let b: Vec<f64> = (0..1000).map(|i| (i + 5000) as f64).collect();
		let d = ks_statistic(&a, &b);
		assert!(
			d > 0.9,
			"completely disjoint samples must have D ≈ 1, got {d}"
		);
	}

	#[test]
	fn test_ks_statistic_shifted() {
		// Two overlapping distributions: shift by a small amount
		let a: Vec<f64> = (0..500).map(|i| i as f64).collect();
		let b: Vec<f64> = (200..700).map(|i| i as f64).collect();
		let d = ks_statistic(&a, &b);
		assert!(d > 0.0, "different distributions must have D > 0, got {d}");
	}

	#[test]
	fn test_ks_critical_value() {
		let crit = ks_critical_value_005(1000, 1000);
		// D_crit ≈ 1.36 * sqrt(2000 / 1_000_000) ≈ 1.36 * 0.0447 ≈ 0.0608
		assert!(
			(crit - 0.0608).abs() < 0.001,
			"unexpected critical value: {crit}"
		);
	}

	#[test]
	fn test_ks_test_passes_identical() {
		let a: Vec<f64> = (0..500).map(|i| i as f64).collect();
		let b: Vec<f64> = (0..500).map(|i| i as f64).collect();
		assert!(ks_test_passes(&a, &b), "identical samples must pass");
	}

	#[test]
	fn test_ks_test_rejects_different() {
		let a: Vec<f64> = (0..500).map(|i| i as f64).collect();
		let b: Vec<f64> = (0..500).map(|i| (i + 500) as f64).collect();
		assert!(!ks_test_passes(&a, &b), "disjoint samples must be rejected");
	}

	// ── Chi-squared test ───────────────────────────────────────────────────

	#[test]
	fn test_chi_squared_identical() {
		let mut obs = [0u64; 256];
		let mut exp = [0u64; 256];
		for i in 0..256 {
			obs[i] = 100;
			exp[i] = 100;
		}
		let stat = chi_squared_byte_test(&obs, &exp);
		assert!(
			stat < 1e-10,
			"identical frequencies must give χ² ≈ 0, got {stat}"
		);
	}

	#[test]
	fn test_chi_squared_different() {
		let mut obs = [0u64; 256];
		let mut exp = [0u64; 256];
		obs[0] = 1000;
		exp[0] = 100;
		for i in 1..256 {
			obs[i] = 100;
			exp[i] = 100;
		}
		let stat = chi_squared_byte_test(&obs, &exp);
		assert!(
			stat > 1000.0,
			"large deviation must give large χ², got {stat}"
		);
	}

	#[test]
	fn test_byte_frequencies() {
		let data = b"hello world";
		let freqs = byte_frequencies(data);
		assert_eq!(freqs[b'h' as usize], 1);
		assert_eq!(freqs[b'l' as usize], 3);
		assert_eq!(freqs[b' ' as usize], 1);

		// Sum of all frequencies equals length
		let total: u64 = freqs.iter().sum();
		assert_eq!(total, data.len() as u64);
	}

	#[test]
	fn test_chi_squared_passes_identical() {
		let mut obs = [0u64; 256];
		let mut exp = [0u64; 256];
		for i in 0..256 {
			obs[i] = 100;
			exp[i] = 100;
		}
		assert!(chi_squared_test_passes(&obs, &exp));
	}

	#[test]
	fn test_chi_squared_rejects_extreme() {
		let mut obs = [0u64; 256];
		let mut exp = [0u64; 256];
		for i in 0..256 {
			obs[i] = if i == 0 { 100_000 } else { 100 };
			exp[i] = 100;
		}
		assert!(!chi_squared_test_passes(&obs, &exp));
	}

	// ── End-to-end stego-like scenario ─────────────────────────────────────

	#[test]
	fn test_stego_scenario() {
		// Build a donor-like tensor (5000 values spread across a realistic range)
		let donor: Vec<f32> = (0..5000)
			.map(|i| (i as f32) / 5000.0 * 20.0 - 10.0)
			.collect();
		let table = build_ecdf_table(&donor).expect("donor must be eligible");
		let (sorted_vals, sorted_idx) = build_sorted_ecdf_table(&donor).unwrap();

		// Simulate an encrypted payload (uniform random looking bytes)
		let key = [0xDEu8; 32];
		let mut rng = PaddingRng::new(&key);
		let payload = rng.generate(500);

		// Encode
		let encoded = encode_chunk(&payload, &table);
		assert_eq!(encoded.len(), payload.len());

		// Decode
		let decoded = decode_chunk(&encoded, &sorted_vals, &sorted_idx);
		assert_eq!(decoded, payload, "stego roundtrip must be lossless");

		// K-S test: encoded values should be statistically similar to donor
		let encoded_f64: Vec<f64> = encoded.iter().map(|&f| f as f64).collect();
		let donor_f64: Vec<f64> = donor.iter().map(|&f| f as f64).collect();

		// They should pass the K-S test at α=0.05
		assert!(
			ks_test_passes(&encoded_f64, &donor_f64),
			"encoded chunk must be statistically similar to donor"
		);
	}

	#[test]
	fn test_ecdf_table_with_constant_values_returns_none() {
		// All values identical → only 1 distinct value
		let donor: Vec<f32> = vec![42.0; 10_000];
		assert!(
			build_ecdf_table(&donor).is_none(),
			"constant-valued tensor must be ineligible"
		);
	}

	#[test]
	#[should_panic(expected = "donor tensor must not be empty")]
	fn test_empty_donor_panics() {
		build_ecdf_table(&[]);
	}

	#[test]
	fn test_padding_rng_uniformity() {
		let key = [0x99u8; 32];
		let mut rng = PaddingRng::new(&key);
		let data = rng.generate(100_000);

		let freqs = byte_frequencies(&data);
		let expected = (100_000 / 256) as u64;

		// No byte should deviate wildly from uniform
		let expected_f = expected as f64;
		for (byte, &freq) in freqs.iter().enumerate() {
			let ratio = freq as f64 / expected_f;
			assert!(
				(0.5..=1.5).contains(&ratio),
				"byte {byte} frequency {freq} is far from expected {expected} (ratio={ratio:.3})"
			);
		}
	}
}
