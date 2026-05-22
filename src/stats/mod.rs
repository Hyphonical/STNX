//! Statistical analysis and ECDF table construction for Constellation Encoding.
//!
//! Implements the order-statistic lookup table builder, two-sample K–S test,
//! chi-squared byte-frequency test, padding CSPRNG (PLAN.md §§4.2, 10.2), and
//! the multiset permutation encoder/decoder for INT8/UINT8 donors (INT8.md
//! §§2, 4, 10).

use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
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
pub fn chi_squared_byte_test(obs: &[u64], exp: &[f64]) -> f64 {
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
		let e = exp[i];
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
pub fn chi_squared_test_passes(obs: &[u64], exp: &[f64]) -> bool {
	let statistic = chi_squared_byte_test(obs, exp);
	statistic <= CHI_SQUARED_CRITICAL_005
}

// ---------------------------------------------------------------------------
// Empirical entropy  (INT8.md §5.2)
// ---------------------------------------------------------------------------

/// Compute the empirical Shannon entropy of a discrete distribution.
///
/// Returns 0.0 if `n == 0`.
pub fn empirical_entropy(counts: &[usize; 256], n: usize) -> f64 {
	if n == 0 {
		return 0.0;
	}
	let nf = n as f64;
	let mut h = 0.0;
	for &c in counts {
		if c > 0 {
			let p = c as f64 / nf;
			h -= p * p.log2();
		}
	}
	h
}

/// Conservative capacity estimate for an INT8/UINT8 donor.
///
/// `C_safe = floor((n * H - 256 * log2(n) - 64) / 8)`  (INT8.md §5.3)
pub fn int8_safe_capacity(n: usize, entropy: f64) -> usize {
	let penalty = 256.0 * (n as f64).log2() + 64.0;
	let bits = n as f64 * entropy - penalty;
	if bits.is_sign_negative() || bits.is_nan() {
		0
	} else {
		(bits / 8.0).floor() as usize
	}
}

// ---------------------------------------------------------------------------
// Fenwick tree for adaptive symbol counts  (INT8.md §10.1)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Fenwick {
	tree: [i64; 257], // 1-indexed
}

impl Fenwick {
	pub fn new() -> Self {
		Self { tree: [0; 257] }
	}

	/// Increment count at index `i` (0-based) by `delta`.
	pub fn add(&mut self, mut i: usize, delta: i64) {
		i += 1;
		while i <= 256 {
			self.tree[i] += delta;
			i += i & i.wrapping_neg();
		}
	}

	/// Inclusive prefix sum [0..i] (0-based).
	pub fn sum(&self, i: usize) -> i64 {
		let mut idx = i + 1;
		let mut res = 0i64;
		while idx > 0 {
			res += self.tree[idx];
			idx -= idx & idx.wrapping_neg();
		}
		res
	}

	/// Total sum of all 256 symbols.
	pub fn total(&self) -> i64 {
		self.sum(255)
	}

	/// Find the largest index such that `prefix_sum(idx) <= target`.
	///
	/// Returns a 0-based index.
	pub fn find(&self, mut target: i64) -> usize {
		let mut idx = 0usize;

		if self.tree[256] <= target {
			idx = 256;
			target -= self.tree[256];
		}
		let mut t = idx + 128;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 64;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 32;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 16;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 8;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 4;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 2;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
		}
		t = idx + 1;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
		}

		idx.saturating_sub(1)
	}

	/// Find the largest index such that `prefix_sum(idx) <= target`,
	/// returning both the 0-based index and the *inclusive* prefix sum
	/// `sum(idx)`.  This avoids the caller making a second O(log N)
	/// `sum(idx)` call just to compute the frequency.
	pub fn find_with_cum(&self, mut target: i64) -> (usize, u64) {
		let mut idx = 0usize;
		let mut cum = 0i64;

		if self.tree[256] <= target {
			idx = 256;
			target -= self.tree[256];
			cum += self.tree[256];
		}
		let mut t = idx + 128;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 64;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 32;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 16;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 8;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 4;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 2;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			target -= self.tree[t];
			cum += self.tree[t];
		}
		t = idx + 1;
		if t <= 256 && self.tree[t] <= target {
			idx = t;
			cum += self.tree[t];
		}

		// idx is 1-based inclusive index; `cum` is `sum(idx)`.
		(idx.saturating_sub(1), cum as u64)
	}
}

impl Default for Fenwick {
	fn default() -> Self {
		Self::new()
	}
}

// ---------------------------------------------------------------------------
// 64-bit range encoder (extraction: symbols → bits)  (INT8.md §10.2)
// ---------------------------------------------------------------------------

pub struct RangeEncoder {
	low: u64,
	range: u64,
	out: Vec<u8>,
}

impl RangeEncoder {
	pub fn new() -> Self {
		Self {
			low: 0,
			range: u64::MAX,
			out: Vec::new(),
		}
	}

	/// Encode a symbol with cumulative count `cum`, frequency `freq`,
	/// given `total` remaining symbols.
	pub fn encode(&mut self, cum: u64, freq: u64, total: u64) {
		let r = (self.range as u128) / (total as u128);
		self.low += (r * (cum as u128)) as u64;
		self.range = (r * (freq as u128)) as u64;

		while self.range < (1u64 << 56) {
			self.out.push((self.low >> 56) as u8);
			self.low <<= 8;
			self.range <<= 8;
		}
	}

	/// Flush remaining state: write 8 final bytes.
	pub fn finish(mut self) -> Vec<u8> {
		for _ in 0..8 {
			self.out.push((self.low >> 56) as u8);
			self.low <<= 8;
		}
		self.out
	}
}

impl Default for RangeEncoder {
	fn default() -> Self {
		Self::new()
	}
}

// ---------------------------------------------------------------------------
// 64-bit range decoder (injection: bits → symbols)  (INT8.md §10.3)
// ---------------------------------------------------------------------------

pub struct RangeDecoder<'a> {
	low: u64,
	range: u64,
	code: u64,
	src: &'a [u8],
	pos: usize,
}

impl<'a> RangeDecoder<'a> {
	pub fn new(src: &'a [u8]) -> Self {
		let mut code = 0u64;
		let mut pos = 0;
		for _ in 0..8 {
			code = (code << 8) | (*src.get(pos).unwrap_or(&0) as u64);
			pos += 1;
		}
		Self {
			low: 0,
			range: u64::MAX,
			code,
			src,
			pos,
		}
	}

	/// Given `total`, return the scaled value used to find the symbol.
	pub fn get_scaled(&self, total: u64) -> u64 {
		(((self.code - self.low) as u128) * (total as u128) / (self.range as u128)) as u64
	}

	/// Consume a symbol with cumulative count `cum`, frequency `freq`,
	/// given `total` remaining symbols.
	pub fn decode(&mut self, cum: u64, freq: u64, total: u64) {
		let r = (self.range as u128) / (total as u128);
		self.low += (r * (cum as u128)) as u64;
		self.range = (r * (freq as u128)) as u64;

		while self.range < (1u64 << 56) {
			self.code = (self.code << 8) | (*self.src.get(self.pos).unwrap_or(&0) as u64);
			self.low <<= 8;
			self.range <<= 8;
			self.pos += 1;
		}
	}

	/// Bytes consumed from the source so far.
	pub fn bytes_consumed(&self) -> usize {
		self.pos
	}
}

// ---------------------------------------------------------------------------
// Bitstream cursor for injection  (INT8.md §10.5)
// ---------------------------------------------------------------------------

pub struct BitstreamCursor {
	payload: Vec<u8>,
	pad: ChaCha20Rng,
	byte_pos: usize,
}

impl BitstreamCursor {
	pub fn new(payload: Vec<u8>, pad_seed: &[u8; 32]) -> Self {
		// Derive ChaCha20 seed from the pad subkey
		let mut seed = [0u8; 32];
		let mut hasher = Sha256::new();
		hasher.update(b"stnx.int8.bitstream");
		hasher.update(pad_seed);
		let hash = hasher.finalize();
		seed.copy_from_slice(&hash);
		let pad = ChaCha20Rng::from_seed(seed);
		Self {
			payload,
			pad,
			byte_pos: 0,
		}
	}

	/// Get the next byte from the cursors. Falls back to CSPRNG padding
	/// once the payload is exhausted.
	pub fn next_byte(&mut self) -> u8 {
		if self.byte_pos < self.payload.len() {
			let b = self.payload[self.byte_pos];
			self.byte_pos += 1;
			b
		} else {
			use rand::Rng;
			let mut buf = [0u8; 1];
			self.pad.fill_bytes(&mut buf);
			buf[0]
		}
	}

	/// Bulk-fill `buf` with remaining payload bytes first, then CSPRNG
	/// padding.  Much faster than calling `next_byte` in a loop because
	/// it avoids per-byte RNG sealing.
	pub fn fill_buffer(&mut self, buf: &mut [u8]) {
		let remaining = self.payload.len().saturating_sub(self.byte_pos);
		let copy = remaining.min(buf.len());
		buf[..copy].copy_from_slice(&self.payload[self.byte_pos..self.byte_pos + copy]);
		self.byte_pos += copy;
		if copy < buf.len() {
			use rand::Rng;
			self.pad.fill_bytes(&mut buf[copy..]);
		}
	}

	pub fn remaining_payload(&self) -> usize {
		self.payload.len().saturating_sub(self.byte_pos)
	}
}

// ---------------------------------------------------------------------------
// Block-level INT8/UINT8 encode / decode  (INT8.md §10.4)
// ---------------------------------------------------------------------------

/// Encode a block of INT8/UINT8 indices into a bitstream.
///
/// `symbols`: the symbol indices (0..255).
/// `counts`: the donor histogram.
/// Returns the compressed bitstream.
pub fn encode_int8_block(symbols: &[u8], counts: &[usize; 256]) -> Vec<u8> {
	let mut fenwick = Fenwick::new();
	let mut flat_counts = [0i64; 256];
	for (i, &c) in counts.iter().enumerate() {
		fenwick.add(i, c as i64);
		flat_counts[i] = c as i64;
	}

	let mut enc = RangeEncoder::new();
	let mut total_symbols = fenwick.total() as u64;
	for &sym in symbols {
		let idx = sym as usize;
		let freq = flat_counts[idx] as u64;
		let cum = fenwick.sum(idx) as u64 - freq;
		let total = total_symbols;
		enc.encode(cum, freq, total);
		fenwick.add(idx, -1);
		flat_counts[idx] -= 1;
		total_symbols -= 1;
	}
	enc.finish()
}

/// Decode a block of INT8/UINT8 indices from a bitstream.
///
/// `bitstream`: the compressed data from `encode_int8_block`.
/// `counts`: the donor histogram.
/// `n`: number of symbols to decode.
/// Returns the decoded symbol indices.
pub fn decode_int8_block(bitstream: &[u8], counts: &[usize; 256], n: usize) -> Vec<u8> {
	let mut fenwick = Fenwick::new();
	for (i, &c) in counts.iter().enumerate() {
		fenwick.add(i, c as i64);
	}

	let mut dec = RangeDecoder::new(bitstream);
	let mut out = Vec::with_capacity(n);
	// Keep flat counts for O(1) frequency lookup — avoids a second Fenwick sum call
	let mut flat_counts = [0i64; 256];
	for (i, &c) in counts.iter().enumerate() {
		flat_counts[i] = c as i64;
	}
	let mut total_symbols = fenwick.total() as u64;
	for _ in 0..n {
		let total = total_symbols;
		let scaled = dec.get_scaled(total);
		let (idx, sum_cum) = fenwick.find_with_cum(scaled as i64);
		// sum_cum is inclusive prefix sum up to idx
		let freq = flat_counts[idx] as u64;
		let cum = sum_cum - freq;
		dec.decode(cum, freq, total);
		fenwick.add(idx, -1);
		flat_counts[idx] -= 1;
		total_symbols -= 1;
		out.push(idx as u8);
	}
	out
}

// ---------------------------------------------------------------------------
// Smart bitstream: encode/decode with hybrid ECDF / multiset dispatch
// ---------------------------------------------------------------------------

/// Encode payload bytes for a given donor.
///
/// FP32/FP16: uses `encode_chunk`.
/// INT8/UINT8: uses the multiset permutation decoder (decodes bits to symbols).
///
/// `donor_scalar_count`: number of elements the stego tensor will have.
/// `remaining_payload`: bytes remaining in the payload.
/// Returns `(stego_values, bytes_consumed_from_payload)`.
pub fn encode_for_donor(
	chunk_bytes: &[u8],
	pad_seed: &[u8; 32],
	data_type: i32,
	ecdf_table: &[f32; 256],
	counts: Option<&[usize; 256]>,
	scalar_count: usize,
) -> Vec<u8> {
	match data_type {
		crate::proto::helpers::DT_FLOAT | crate::proto::helpers::DT_FLOAT16 => {
			// ECDF path: 1 byte → 1 float (PLAN.md §4.2)
			let encoded = encode_chunk(chunk_bytes, ecdf_table);
			// Pad if needed
			let combined = if encoded.len() < scalar_count {
				let pad_len = scalar_count - encoded.len();
				let mut rng = PaddingRng::new(pad_seed);
				let pad_bytes = rng.generate(pad_len);
				let pad_encoded = encode_chunk(&pad_bytes, ecdf_table);
				let mut c = encoded;
				c.extend_from_slice(&pad_encoded);
				c
			} else {
				encoded
			};
			serialize_floats(&combined, data_type)
		}
		crate::proto::helpers::DT_INT8 | crate::proto::helpers::DT_UINT8 => {
			// Multiset permutation path (INT8.md §4)
			let counts_ref = counts.expect("counts required for INT8/UINT8");
			// Prepare bitstream: payload bytes + fallback to K_pad CSPRNG
			let mut cursor = BitstreamCursor::new(chunk_bytes.to_vec(), pad_seed);
			let n = scalar_count;

			// Bulk-allocate and fill the bitstream in one shot
			let bitstream_len = 8 + 2 * n;
			let mut bitstream = vec![0u8; bitstream_len];
			cursor.fill_buffer(&mut bitstream);

			let indices = decode_int8_block(&bitstream, counts_ref, n);

			// Convert indices back to raw bytes
			let raw: Vec<u8> = match data_type {
				crate::proto::helpers::DT_INT8 => indices
					.iter()
					.map(|&idx| (idx as i16 - 128) as i8 as u8)
					.collect(),
				_ => indices,
			};
			raw
		}
		_ => unreachable!("unsupported data_type in encode_for_donor"),
	}
}

/// Decode stego raw_data bytes for a given donor back to payload bytes.
///
/// FP32/FP16: uses `decode_chunk`.
/// INT8/UINT8: uses the multiset permutation encoder (symbols → bits).
pub fn decode_for_donor(
	raw_data: &[u8],
	data_type: i32,
	sorted_vals: &[f32; 256],
	sorted_idx: &[u8; 256],
	counts: Option<&[usize; 256]>,
	_scalar_count: usize,
) -> Vec<u8> {
	match data_type {
		crate::proto::helpers::DT_FLOAT | crate::proto::helpers::DT_FLOAT16 => {
			let floats = deserialize_floats(raw_data, data_type);
			decode_chunk(&floats, sorted_vals, sorted_idx)
		}
		crate::proto::helpers::DT_INT8 | crate::proto::helpers::DT_UINT8 => {
			let counts_ref = counts.expect("counts required for INT8/UINT8");
			// Convert raw bytes to indices
			let indices: Vec<u8> = match data_type {
				crate::proto::helpers::DT_INT8 => raw_data
					.iter()
					.map(|&b| ((b as i8) as i16 + 128) as u8)
					.collect(),
				_ => raw_data.to_vec(),
			};
			encode_int8_block(&indices, counts_ref)
		}
		_ => unreachable!("unsupported data_type in decode_for_donor"),
	}
}

/// Serialize f32 values to raw bytes for a given data_type.
fn serialize_floats(values: &[f32], data_type: i32) -> Vec<u8> {
	match data_type {
		crate::proto::helpers::DT_FLOAT => {
			let mut buf = Vec::with_capacity(values.len() * 4);
			for &v in values {
				buf.extend_from_slice(&v.to_le_bytes());
			}
			buf
		}
		crate::proto::helpers::DT_FLOAT16 => {
			use crate::proto::helpers::f32_to_f16;
			let mut buf = Vec::with_capacity(values.len() * 2);
			for &v in values {
				buf.extend_from_slice(&f32_to_f16(v).to_le_bytes());
			}
			buf
		}
		_ => unreachable!(),
	}
}

/// Deserialize raw bytes to f32 values for a given data_type.
fn deserialize_floats(raw: &[u8], data_type: i32) -> Vec<f32> {
	match data_type {
		crate::proto::helpers::DT_FLOAT => raw
			.chunks_exact(4)
			.map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
			.collect(),
		crate::proto::helpers::DT_FLOAT16 => {
			use crate::proto::helpers::f16_to_f32;
			raw.chunks_exact(2)
				.map(|chunk| {
					let bits = u16::from_le_bytes(chunk.try_into().unwrap());
					f16_to_f32(bits)
				})
				.collect()
		}
		_ => unreachable!(),
	}
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
		let mut exp = [0.0f64; 256];
		for i in 0..256 {
			obs[i] = 100;
			exp[i] = 100.0;
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
		let mut exp = [0.0f64; 256];
		obs[0] = 1000;
		exp[0] = 100.0;
		for i in 1..256 {
			obs[i] = 100;
			exp[i] = 100.0;
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
		let mut exp = [0.0f64; 256];
		for i in 0..256 {
			obs[i] = 100;
			exp[i] = 100.0;
		}
		assert!(chi_squared_test_passes(&obs, &exp));
	}

	#[test]
	fn test_chi_squared_rejects_extreme() {
		let mut obs = [0u64; 256];
		let mut exp = [0.0f64; 256];
		for i in 0..256 {
			obs[i] = if i == 0 { 100_000 } else { 100 };
			exp[i] = 100.0;
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
