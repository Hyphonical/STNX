//! Constellation Encoding engine for ONNX steganography.
//!
//! Implements the stego tensor name generator, donor selector (Natural Ratio),
//! profiling, injection, and extraction orchestration as specified in
//! PLAN.md §§4.4, 6, 7.1–7.3 and INT8.md §§3, 4, 9.

use crate::proto::helpers::{self, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, EligibleTensor};
use owo_colors::OwoColorize;
use sha2::{Digest, Sha256};
use std::path::Path;

// ---------------------------------------------------------------------------
// Helpers  (CLI formatting — PLAN.md §7.1, docs/cli-design.ps1)
// ---------------------------------------------------------------------------

/// Format a number with space-separated thousands (e.g. `43 521 024`).
fn fmt_count(n: usize) -> String {
	let s = n.to_string();
	let mut result = String::with_capacity(s.len() + s.len() / 3);
	for (i, ch) in s.char_indices() {
		if i > 0 && (s.len() - i).is_multiple_of(3) {
			result.push(' ');
		}
		result.push(ch);
	}
	result
}

// ---------------------------------------------------------------------------
// Name generation  (PLAN.md §4.4)
// ---------------------------------------------------------------------------

/// Family of plausible stego tensor name templates.
const NAME_FAMILIES: &[&str] = &[
	"_ema.shadow.{hex}.{idx:04}",
	"_optim.exp_avg_sq.{hex}.{idx:04}",
	"_optim.exp_avg.{hex}.{idx:04}",
	"lora_B.{hex}.{idx:04}",
	"rotary_emb.inv_freq_ext_{hex}.{idx:04}",
];

/// Deterministic CSPRNG for generating stego tensor names.
///
/// Seeded by `K_name`.  Produces an infinite stream of plausible tensor names
/// that are checked for collision against existing model tensor names.
pub struct NameGenerator {
	key: [u8; 32],
	counter: u64,
	family_index: usize,
	seq_index: u64,
	buffer: [u8; 32],
	pos: usize,
}

impl NameGenerator {
	/// Create a new name generator from the `K_name` subkey.
	pub fn new(key: &[u8; 32]) -> Self {
		let mut rng = Self {
			key: *key,
			counter: 0,
			family_index: 0,
			seq_index: 0,
			buffer: [0u8; 32],
			pos: 32,
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

	fn next_u8(&mut self) -> u8 {
		if self.pos >= 32 {
			self.refill();
		}
		let b = self.buffer[self.pos];
		self.pos += 1;
		b
	}

	fn next_u32(&mut self) -> u32 {
		let mut bytes = [0u8; 4];
		for b in &mut bytes {
			*b = self.next_u8();
		}
		u32::from_le_bytes(bytes)
	}

	/// Generate the next name in the deterministic sequence.
	fn generate_name(&mut self) -> String {
		let family = NAME_FAMILIES[self.family_index];
		self.family_index = (self.family_index + 1) % NAME_FAMILIES.len();

		let hex_suffix = format!("{:08x}", self.next_u32());
		let idx = self.seq_index;
		self.seq_index += 1;

		family
			.replace("{hex}", &hex_suffix)
			.replace("{idx:04}", &format!("{idx:04}"))
	}

	/// Produce the next name that does not collide with `existing_names`.
	pub fn next_name(&mut self, existing_names: &[String]) -> String {
		loop {
			let name = self.generate_name();
			if !existing_names.contains(&name) {
				return name;
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Donor selector  (PLAN.md §6)
// ---------------------------------------------------------------------------

/// Deterministic CSPRNG for selecting donors from the eligible pool.
///
/// Seeded by `K_profile`.  Produces an infinite stream of indices into the
/// eligible tensor list.
pub struct DonorSelector {
	key: [u8; 32],
	counter: u64,
	buffer: [u8; 32],
	pos: usize,
}

impl DonorSelector {
	/// Create a new donor selector from the `K_profile` subkey.
	pub fn new(key: &[u8; 32]) -> Self {
		let mut sel = Self {
			key: *key,
			counter: 0,
			buffer: [0u8; 32],
			pos: 32,
		};
		sel.refill();
		sel
	}

	fn refill(&mut self) {
		let mut hasher = Sha256::new();
		hasher.update(self.key);
		hasher.update(self.counter.to_le_bytes());
		self.buffer = hasher.finalize().into();
		self.counter += 1;
		self.pos = 0;
	}

	fn next_u32(&mut self) -> u32 {
		let mut bytes = [0u8; 4];
		for b in &mut bytes {
			if self.pos >= 32 {
				self.refill();
			}
			*b = self.buffer[self.pos];
			self.pos += 1;
		}
		u32::from_le_bytes(bytes)
	}

	/// Select the next donor index from the eligible pool (uniform random).
	pub fn next_index(&mut self, pool_size: usize) -> usize {
		if pool_size == 0 {
			panic!("donor pool is empty");
		}
		(self.next_u32() as usize) % pool_size
	}

	/// Consume one raw u32 from the CSPRNG without applying a modulus.
	///
	/// Used by biased donor selection to consume a predictable number of
	/// CSPRNG outputs regardless of pool composition.
	pub(crate) fn next_raw_u32(&mut self) -> u32 {
		self.next_u32()
	}
}

// ---------------------------------------------------------------------------
// Natural Ratio donor selection  (INT8.md §3)
// ---------------------------------------------------------------------------

/// Dtype pools for Natural Ratio selection.
struct Pool<'a> {
	tensors: Vec<&'a EligibleTensor>,
	total_elements: usize,
}

impl<'a> Pool<'a> {
	fn weight(&self) -> usize {
		self.total_elements
	}
}

/// Select the next donor using Natural Ratio demographic weighting.
///
/// Each call consumes exactly 1 raw u32 from the CSPRNG, ensuring
/// deterministic replay between injection and extraction.
///
/// Strategy (INT8.md §3.1):
/// 1. Compute Natural Ratio weights from per-dtype eligible elements.
/// 2. Draw a dtype via CSPRNG using the categorical distribution.
/// 3. Select the **largest remaining unselected** tensor from that pool.
/// 4. If a pool is exhausted, renormalize over remaining pools.
///
/// # Panics
///
/// Panics if all pools are empty.
fn select_donor_natural<'a>(
	donor_sel: &mut DonorSelector,
	pools: &mut [Pool<'a>],
) -> &'a EligibleTensor {
	let total_weight: usize = pools.iter().map(|p| p.weight()).sum();
	assert!(total_weight > 0, "all donor pools are empty");

	let choice = (donor_sel.next_raw_u32() as usize) % total_weight;
	let mut cumulative = 0usize;

	// Find which dtype pool was selected
	let pool_idx = pools
		.iter()
		.position(|p| {
			cumulative += p.weight();
			choice < cumulative
		})
		.expect("categorical choice must fall within total weight");

	// Consume the largest remaining tensor from the chosen pool
	let pool = &mut pools[pool_idx];
	if pool.tensors.is_empty() {
		// Pool exhausted; renormalize: zero out and retry
		pool.total_elements = 0;
		return select_donor_natural(donor_sel, pools);
	}

	// Descending-size order means tensors[0] is the largest remaining
	let result = pool.tensors.remove(0);
	// Update pool weight (recompute from remaining tensors)
	pool.total_elements = pool.tensors.iter().map(|t| t.scalar_count).sum();
	result
}

/// Build dtype pools for Natural Ratio selection.
fn build_pools<'a>(eligible: &'a [EligibleTensor]) -> Vec<Pool<'a>> {
	let fp32: Vec<&EligibleTensor> = eligible
		.iter()
		.filter(|e| e.data_type == DT_FLOAT)
		.collect();
	let fp16: Vec<&EligibleTensor> = eligible
		.iter()
		.filter(|e| e.data_type == DT_FLOAT16)
		.collect();
	let int8: Vec<&EligibleTensor> = eligible.iter().filter(|e| e.data_type == DT_INT8).collect();
	let uint8: Vec<&EligibleTensor> = eligible
		.iter()
		.filter(|e| e.data_type == DT_UINT8)
		.collect();

	let mut pools = Vec::new();
	for tensors in [fp32, fp16, int8, uint8] {
		if !tensors.is_empty() {
			let total_elements = tensors.iter().map(|t| t.scalar_count).sum();
			pools.push(Pool {
				tensors,
				total_elements,
			});
		}
	}
	pools
}

// ---------------------------------------------------------------------------
// Profile  (PLAN.md §7.1 + INT8.md §5.4)
// ---------------------------------------------------------------------------

/// Profile a donor ONNX model and return a capacity report string.
pub fn profile(model_path: &Path, alpha: f64) -> Result<String, Box<dyn std::error::Error>> {
	let model = helpers::load_model(model_path)?;
	let eligible = helpers::eligible_initializers(&model);

	if eligible.is_empty() {
		return Err("No eligible tensors found (FP32, FP16, INT8, or UINT8)".into());
	}

	let total_el = helpers::total_eligible_elements(&eligible);
	let max_payload = helpers::max_payload_bytes(total_el, alpha);

	// Count per-dtype
	let macros = |dt: i32| {
		let count = eligible.iter().filter(|e| e.data_type == dt).count();
		let el: usize = eligible
			.iter()
			.filter(|e| e.data_type == dt)
			.map(|e| e.scalar_count)
			.sum();
		(count, el)
	};
	let (fp32_cnt, fp32_el) = macros(DT_FLOAT);
	let (fp16_cnt, fp16_el) = macros(DT_FLOAT16);
	let (int8_cnt, int8_el) = macros(DT_INT8);
	let (uint8_cnt, uint8_el) = macros(DT_UINT8);

	// Disk overhead (INT8.md §8.1)
	let disk_multiplier = if total_el > 0 {
		(4.0 * fp32_el as f64 + 2.0 * fp16_el as f64 + 1.0 * int8_el as f64 + 1.0 * uint8_el as f64)
			/ total_el as f64
	} else {
		0.0
	};

	let projected_payload_mb = (max_payload as f64) / (1024.0 * 1024.0);
	let projected_disk_mb = projected_payload_mb * disk_multiplier;

	// Natural Ratio percentages (INT8.md §3.1)
	let ratio_str = if total_el > 0 {
		format!(
			"FP32 {:.1}% | FP16 {:.1}% | INT8 {:.1}% | UINT8 {:.1}%",
			fp32_el as f64 / total_el as f64 * 100.0,
			fp16_el as f64 / total_el as f64 * 100.0,
			int8_el as f64 / total_el as f64 * 100.0,
			uint8_el as f64 / total_el as f64 * 100.0,
		)
	} else {
		"N/A".to_string()
	};

	let mut lines = vec!["Profile complete.".blue().bold().to_string()];

	if fp32_cnt > 0 {
		lines.push(format!(
			"  {:30} : {} {}",
			"Eligible FP32 tensors".blue(),
			fmt_count(fp32_cnt).white(),
			format!("(total elements: {})", fmt_count(fp32_el)).dimmed(),
		));
	}
	if fp16_cnt > 0 {
		lines.push(format!(
			"  {:30} : {} {}",
			"Eligible FP16 tensors".blue(),
			fmt_count(fp16_cnt).white(),
			format!("(total elements: {})", fmt_count(fp16_el)).dimmed(),
		));
	}
	if int8_cnt > 0 {
		lines.push(format!(
			"  {:30} : {} {}",
			"Eligible INT8 tensors".blue(),
			fmt_count(int8_cnt).white(),
			format!("(total elements: {}, entropy≥4.0)", fmt_count(int8_el)).dimmed(),
		));
	}
	if uint8_cnt > 0 {
		lines.push(format!(
			"  {:30} : {} {}",
			"Eligible UINT8 tensors".blue(),
			fmt_count(uint8_cnt).white(),
			format!("(total elements: {}, entropy≥4.0)", fmt_count(uint8_el)).dimmed(),
		));
	}

	lines.push(format!(
		"  {:30} : {}",
		"Combined eligible elements".blue(),
		fmt_count(total_el).white(),
	));
	lines.push(format!(
		"  {:30} : Natural Ratio (by element)   {}",
		"".blue(),
		ratio_str.white(),
	));
	lines.push(format!(
		"  {:30} : {:.1}x{}",
		"Effective disk overhead".blue(),
		disk_multiplier,
		" payload bytes".white(),
	));
	lines.push(format!(
		"  {:30} : ~{:.1} MB{}",
		format!("Projected capacity @ {:.0}%", alpha * 100.0).blue(),
		projected_payload_mb,
		" payload".white(),
	));
	lines.push(format!(
		"  {:30} : ~{:.1} MB{}",
		"".blue(),
		projected_disk_mb,
		" disk overhead".white(),
	));

	Ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// Inject  (PLAN.md §7.2 + INT8.md §9.2)
// ---------------------------------------------------------------------------

/// Inject a payload into a donor ONNX model.
pub fn inject(
	model_path: &Path,
	payload: &[u8],
	passphrase: &str,
	out_path: &Path,
	alpha: f64,
) -> Result<(), Box<dyn std::error::Error>> {
	let subkeys = crate::crypto::derive_subkeys(passphrase)?;
	let mut model = helpers::load_model(model_path)?;
	let eligible = helpers::eligible_initializers(&model);
	let existing_names = helpers::existing_initializer_names(&model);

	if eligible.is_empty() {
		return Err("No eligible tensors found (FP32, FP16, INT8, or UINT8)".into());
	}

	// ── Capacity check ──
	let total_el = helpers::total_eligible_elements(&eligible);
	let max_payload = helpers::max_payload_bytes(total_el, alpha);
	if payload.len() > max_payload {
		return Err(format!(
			"Payload size {} B exceeds maximum capacity {} B (α·N = {} eligible elements × {:.0}%)",
			payload.len(),
			max_payload,
			total_el,
			alpha * 100.0,
		)
		.into());
	}

	// Build Natural Ratio pools (sorted descending already by eligible_initializers)
	let mut pools = build_pools(&eligible);
	if pools.is_empty() {
		return Err("No eligible tensors found (FP32, FP16, INT8, or UINT8)".into());
	}

	let mut name_gen = NameGenerator::new(&subkeys.name);
	let mut donor_sel = DonorSelector::new(&subkeys.profile);

	let total_payload_bytes = payload.len();
	let mut bytes_consumed = 0usize;
	let mut stego_tensors = Vec::new();

	while bytes_consumed < total_payload_bytes {
		let donor = select_donor_natural(&mut donor_sel, &mut pools);

		match donor.data_type {
			DT_FLOAT | DT_FLOAT16 => {
				// ECDF path
				let table = crate::stats::build_ecdf_table(&donor.sorted_values)
					.ok_or("Donor failed ECDF table construction")?;

				let chunk_size = donor.scalar_count;
				let end = (bytes_consumed + chunk_size).min(payload.len());
				let chunk_bytes = &payload[bytes_consumed..end];

				let encoded = crate::stats::encode_chunk(chunk_bytes, &table);

				let combined = if encoded.len() < chunk_size {
					let mut pad_rng = crate::stats::PaddingRng::new(&subkeys.pad);
					let pad_len = chunk_size - encoded.len();
					let pad_bytes = pad_rng.generate(pad_len);
					let pad_encoded = crate::stats::encode_chunk(&pad_bytes, &table);
					let mut c = encoded;
					c.extend_from_slice(&pad_encoded);
					c
				} else {
					encoded
				};

				let stego_name = name_gen.next_name(&existing_names);
				let stego =
					helpers::build_tensor(&stego_name, &donor.dims, donor.data_type, &combined);

				// ── Verification gates (PLAN.md §10.2) ──
				let num_elements = combined.len();
				let mut pad_rng = crate::stats::PaddingRng::new(&subkeys.pad);
				let synth_bytes = pad_rng.generate(num_elements);
				let synth_encoded = crate::stats::encode_chunk(&synth_bytes, &table);

				let stego_f64: Vec<f64> = combined.iter().map(|&f| f as f64).collect();
				let synth_f64: Vec<f64> = synth_encoded.iter().map(|&f| f as f64).collect();
				let ks_stat = crate::stats::ks_statistic(&stego_f64, &synth_f64);
				let ks_crit = crate::stats::ks_critical_value_005(stego_f64.len(), synth_f64.len());
				if ks_stat > ks_crit {
					return Err(format!(
						"K–S test failed for chunk at donor '{}' (D={:.6}, crit={:.6})",
						donor.name, ks_stat, ks_crit
					)
					.into());
				}

				// Chi-squared byte-frequency test
				let raw = stego.raw_data.as_deref().unwrap_or_default();
				let obs_freqs = crate::stats::byte_frequencies(raw);

				let mut exp_freqs = [0.0f64; 256];
				let expected_count_per_entry = num_elements as f64 / 256.0;
				match donor.data_type {
					DT_FLOAT => {
						for &v in table.iter() {
							for b in v.to_le_bytes() {
								exp_freqs[b as usize] += expected_count_per_entry;
							}
						}
					}
					DT_FLOAT16 => {
						for &v in table.iter() {
							for b in helpers::f32_to_f16(v).to_le_bytes() {
								exp_freqs[b as usize] += expected_count_per_entry;
							}
						}
					}
					_ => unreachable!(),
				}

				let chi2_stat = crate::stats::chi_squared_byte_test(&obs_freqs, &exp_freqs);
				if chi2_stat > crate::stats::CHI_SQUARED_CRITICAL_005 {
					return Err(format!(
						"χ² test failed for chunk at donor '{}' (χ²={:.2}, crit={:.2})",
						donor.name,
						chi2_stat,
						crate::stats::CHI_SQUARED_CRITICAL_005
					)
					.into());
				}

				bytes_consumed = end;
				stego_tensors.push(stego);
			}
			DT_INT8 | DT_UINT8 => {
				// Multiset permutation path (INT8.md §4)
				let chunk_size = donor.scalar_count;
				let end = (bytes_consumed + chunk_size).min(payload.len());
				let chunk_bytes = &payload[bytes_consumed..end];

				let raw = crate::stats::encode_for_donor(
					chunk_bytes,
					&subkeys.pad,
					donor.data_type,
					&[0.0f32; 256], // unused for INT8
					donor.counts.as_ref(),
					chunk_size,
				);

				let stego_name = name_gen.next_name(&existing_names);
				let stego =
					helpers::build_tensor_from_raw(&stego_name, &donor.dims, donor.data_type, raw);

				bytes_consumed = end;
				stego_tensors.push(stego);
			}
			_ => unreachable!("unsupported data_type"),
		};
	}

	// Append stego tensors to model
	if let Some(ref mut graph) = model.graph {
		graph.initializer.extend(stego_tensors);
	} else {
		return Err("Model has no graph".into());
	}

	helpers::save_model(&model, out_path)?;
	Ok(())
}

// ---------------------------------------------------------------------------
// Extract  (PLAN.md §6 + INT8.md §9.3)
// ---------------------------------------------------------------------------

/// Extract a hidden payload from a stego ONNX model.
pub fn extract(
	model_path: &Path,
	passphrase: &str,
) -> Result<(Vec<u8>, String), Box<dyn std::error::Error>> {
	let subkeys = crate::crypto::derive_subkeys(passphrase)?;
	let model = helpers::load_model(model_path)?;

	// Step 1: all initializer names → M
	let all_names = helpers::existing_initializer_names(&model);
	let all_name_set: std::collections::HashSet<&str> =
		all_names.iter().map(|s| s.as_str()).collect();

	// Step 2-3: regenerate the name CSPRNG sequence
	let mut name_gen = NameGenerator::new(&subkeys.name);
	let mut stego_set: Vec<String> = Vec::new();
	let mut used_names: Vec<String> = Vec::new();

	for _ in 0..all_names.len().max(1) * 4 {
		let candidate = name_gen.next_name(&used_names);

		if all_name_set.contains(candidate.as_str()) {
			stego_set.push(candidate.clone());
		}
		used_names.push(candidate);

		if stego_set.len() >= all_names.len() {
			break;
		}
	}

	if stego_set.is_empty() {
		return Err("No stego tensors found — wrong passphrase?".into());
	}

	// Step 3 (continued): donor pool R = M \ S
	let all_eligible = helpers::eligible_initializers(&model);
	let donor_vec: Vec<EligibleTensor> = all_eligible
		.into_iter()
		.filter(|e| !stego_set.contains(&e.name))
		.collect();

	if donor_vec.is_empty() {
		return Err("Donor pool is empty".into());
	}

	let mut pools = build_pools(&donor_vec);
	if pools.is_empty() {
		return Err("Donor pool has no eligible tensors".into());
	}

	// Step 4-5: decode chunks using Natural Ratio donor sequence
	let mut decoded_bytes = Vec::new();
	let mut donor_sel = DonorSelector::new(&subkeys.profile);

	for stego_name in &stego_set {
		let donor = select_donor_natural(&mut donor_sel, &mut pools);

		let stego_tensor = model
			.graph
			.as_ref()
			.and_then(|g| {
				g.initializer
					.iter()
					.find(|t| t.name.as_deref() == Some(stego_name))
			})
			.ok_or("Stego tensor not found in model")?;

		match donor.data_type {
			DT_FLOAT | DT_FLOAT16 => {
				let (sorted_vals, sorted_idx) =
					crate::stats::build_sorted_ecdf_table(&donor.sorted_values)
						.ok_or("Donor failed sorted ECDF table construction")?;

				let floats = helpers::extract_scalars(stego_tensor)?;
				let chunk = crate::stats::decode_chunk(&floats, &sorted_vals, &sorted_idx);
				decoded_bytes.extend_from_slice(&chunk);
			}
			DT_INT8 | DT_UINT8 => {
				let raw_data = stego_tensor
					.raw_data
					.as_deref()
					.ok_or("Stego tensor has no raw_data")?;
				let chunk = crate::stats::decode_for_donor(
					raw_data,
					donor.data_type,
					&[0.0; 256],
					&[0; 256],
					donor.counts.as_ref(),
					donor.scalar_count,
				);
				decoded_bytes.extend_from_slice(&chunk);
			}
			_ => unreachable!("unsupported data_type"),
		}
	}

	// Step 6: decrypt
	let (file_data, filename) = crate::crypto::decrypt_payload(&decoded_bytes, &subkeys.enc)
		.map_err(|e| format!("Decryption failed: {e}"))?;
	Ok((file_data, filename))
}

// ---------------------------------------------------------------------------
// Verify  (PLAN.md §8, §10 + INT8.md §7)
// ---------------------------------------------------------------------------

/// Result of verifying a single stego chunk.
#[derive(Debug, Clone)]
pub struct ChunkReport {
	pub stego_name: String,
	pub donor_name: String,
	pub donor_dtype: i32,
	pub ks_pass: bool,
	pub ks_stat: f64,
	pub ks_crit: f64,
	pub chi2_pass: bool,
	pub chi2_stat: f64,
	pub chi2_crit: f64,
	pub exact_histogram_match: bool,
}

/// Summary of a full verification run.
#[derive(Debug)]
pub struct VerifyReport {
	pub chunks: Vec<ChunkReport>,
	pub all_pass: bool,
	pub total_chunks: usize,
	pub failed_chunks: usize,
}

/// Verify the statistical integrity of a stego ONNX model.
pub fn verify(
	model_path: &Path,
	passphrase: &str,
) -> Result<VerifyReport, Box<dyn std::error::Error>> {
	let subkeys = crate::crypto::derive_subkeys(passphrase)?;
	let model = helpers::load_model(model_path)?;

	// ── Discover stego tensors ──
	let all_names = helpers::existing_initializer_names(&model);
	let all_name_set: std::collections::HashSet<&str> =
		all_names.iter().map(|s| s.as_str()).collect();

	let mut name_gen = NameGenerator::new(&subkeys.name);
	let mut stego_set: Vec<String> = Vec::new();
	let mut used_names: Vec<String> = Vec::new();

	for _ in 0..all_names.len().max(1) * 4 {
		let candidate = name_gen.next_name(&used_names);
		if all_name_set.contains(candidate.as_str()) {
			stego_set.push(candidate.clone());
		}
		used_names.push(candidate);
		if stego_set.len() >= all_names.len() {
			break;
		}
	}

	if stego_set.is_empty() {
		return Err("No stego tensors found — wrong passphrase?".into());
	}

	// ── Build donor pool R = M \ S ──
	let all_eligible = helpers::eligible_initializers(&model);
	let donor_vec: Vec<EligibleTensor> = all_eligible
		.into_iter()
		.filter(|e| !stego_set.contains(&e.name))
		.collect();

	if donor_vec.is_empty() {
		return Err("Donor pool is empty".into());
	}

	let mut pools = build_pools(&donor_vec);

	// ── Verify each chunk ──
	let graph = model.graph.as_ref().ok_or("Model has no graph")?;
	let mut donor_sel = DonorSelector::new(&subkeys.profile);
	let mut chunks = Vec::new();
	let chi2_crit = crate::stats::CHI_SQUARED_CRITICAL_005;

	for stego_name in &stego_set {
		let donor = select_donor_natural(&mut donor_sel, &mut pools);

		let stego_tensor = graph
			.initializer
			.iter()
			.find(|t| t.name.as_deref() == Some(stego_name))
			.ok_or_else(|| format!("Stego tensor '{}' not found", stego_name))?;

		match donor.data_type {
			DT_FLOAT | DT_FLOAT16 => {
				let table = crate::stats::build_ecdf_table(&donor.sorted_values)
					.ok_or_else(|| format!("Donor '{}' failed ECDF", donor.name))?;

				let stego_floats = helpers::extract_scalars(stego_tensor)?;
				let num_elements = stego_floats.len();

				let mut pad_rng = crate::stats::PaddingRng::new(&subkeys.pad);
				let synth_bytes = pad_rng.generate(num_elements);
				let synth_encoded = crate::stats::encode_chunk(&synth_bytes, &table);

				let stego_f64: Vec<f64> = stego_floats.iter().map(|&f| f as f64).collect();
				let synth_f64: Vec<f64> = synth_encoded.iter().map(|&f| f as f64).collect();
				let ks_stat = crate::stats::ks_statistic(&stego_f64, &synth_f64);
				let ks_crit = crate::stats::ks_critical_value_005(stego_f64.len(), synth_f64.len());
				let ks_pass = ks_stat <= ks_crit;

				let raw_data = stego_tensor.raw_data.as_deref().unwrap_or_default();
				let obs_freqs = crate::stats::byte_frequencies(raw_data);

				let mut exp_freqs = [0.0f64; 256];
				let expected_count_per_entry = num_elements as f64 / 256.0;
				match donor.data_type {
					DT_FLOAT => {
						for &v in table.iter() {
							for b in v.to_le_bytes() {
								exp_freqs[b as usize] += expected_count_per_entry;
							}
						}
					}
					DT_FLOAT16 => {
						for &v in table.iter() {
							for b in helpers::f32_to_f16(v).to_le_bytes() {
								exp_freqs[b as usize] += expected_count_per_entry;
							}
						}
					}
					_ => unreachable!(),
				}

				let chi2_stat = crate::stats::chi_squared_byte_test(&obs_freqs, &exp_freqs);
				let chi2_pass = chi2_stat <= chi2_crit;

				chunks.push(ChunkReport {
					stego_name: stego_name.clone(),
					donor_name: donor.name.clone(),
					donor_dtype: donor.data_type,
					ks_pass,
					ks_stat,
					ks_crit,
					chi2_pass,
					chi2_stat,
					chi2_crit,
					exact_histogram_match: false,
				});
			}
			DT_INT8 | DT_UINT8 => {
				// Exact histogram match (INT8.md §7.2)
				let counts_ref = donor.counts.as_ref().expect("counts required");
				let raw_data = stego_tensor
					.raw_data
					.as_deref()
					.ok_or("Stego tensor missing raw_data")?;

				let indices: Vec<u8> = match donor.data_type {
					DT_INT8 => raw_data
						.iter()
						.map(|&b| ((b as i8) as i16 + 128) as u8)
						.collect(),
					_ => raw_data.to_vec(),
				};

				let mut obs_counts = [0usize; 256];
				for &idx in &indices {
					obs_counts[idx as usize] += 1;
				}

				let exact_match = counts_ref == &obs_counts;

				// Entropy match
				let donor_entropy = crate::stats::empirical_entropy(counts_ref, donor.scalar_count);
				let stego_entropy = crate::stats::empirical_entropy(&obs_counts, indices.len());
				let entropy_ok = (donor_entropy - stego_entropy).abs() < 1e-6;

				let histogram_pass = exact_match && entropy_ok;

				chunks.push(ChunkReport {
					stego_name: stego_name.clone(),
					donor_name: donor.name.clone(),
					donor_dtype: donor.data_type,
					ks_pass: histogram_pass, // K-S not applicable for discrete
					ks_stat: 0.0,
					ks_crit: 0.0,
					chi2_pass: histogram_pass,
					chi2_stat: 0.0,
					chi2_crit: 0.0,
					exact_histogram_match: histogram_pass,
				});
			}
			_ => unreachable!(),
		}
	}

	let failed_chunks = chunks.iter().filter(|c| !c.ks_pass || !c.chi2_pass).count();

	Ok(VerifyReport {
		all_pass: failed_chunks == 0,
		total_chunks: chunks.len(),
		failed_chunks,
		chunks,
	})
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::proto::helpers::DT_FLOAT;

	#[test]
	fn test_name_generator_deterministic() {
		let key = [0xABu8; 32];
		let existing: Vec<String> = vec![];

		let mut gen1 = NameGenerator::new(&key);
		let mut gen2 = NameGenerator::new(&key);

		for _ in 0..20 {
			let a = gen1.next_name(&existing);
			let b = gen2.next_name(&existing);
			assert_eq!(a, b, "name generator must be deterministic");
		}
	}

	#[test]
	fn test_name_generator_collision_avoidance() {
		let key = [0xCDu8; 32];
		let mut generator = NameGenerator::new(&key);

		let first = generator.next_name(&[]);
		let existing = vec![first.clone()];

		let second = generator.next_name(&existing);
		assert_ne!(first, second, "must skip colliding names");
	}

	#[test]
	fn test_name_generator_unique_names() {
		let key = [0xEFu8; 32];
		let mut generator = NameGenerator::new(&key);
		let mut names: Vec<String> = Vec::new();

		for _ in 0..100 {
			let name = generator.next_name(&names);
			assert!(!names.contains(&name), "duplicate name generated");
			names.push(name);
		}
	}

	#[test]
	fn test_donor_selector_deterministic() {
		let key = [0x01u8; 32];
		let mut sel1 = DonorSelector::new(&key);
		let mut sel2 = DonorSelector::new(&key);

		for _ in 0..20 {
			assert_eq!(sel1.next_index(100), sel2.next_index(100));
		}
	}

	#[test]
	fn test_donor_selector_respects_pool_size() {
		let key = [0x02u8; 32];
		let mut sel = DonorSelector::new(&key);

		for _ in 0..1000 {
			let idx = sel.next_index(50);
			assert!(idx < 50, "index {idx} out of range");
		}
	}

	#[test]
	fn test_selector_different_seeds() {
		let key_a = [0xAAu8; 32];
		let key_b = [0xBBu8; 32];
		let mut sel_a = DonorSelector::new(&key_a);
		let mut sel_b = DonorSelector::new(&key_b);

		let mut same = true;
		for _ in 0..10 {
			if sel_a.next_index(100) != sel_b.next_index(100) {
				same = false;
				break;
			}
		}
		assert!(!same, "different seeds should produce different sequences");
	}

	#[test]
	fn test_natural_ratio_deterministic() {
		let key = [0x42u8; 32];

		// Build two identical donor pools
		let donors_a = vec![
			EligibleTensor {
				name: "a".into(),
				dims: vec![1024],
				data_type: DT_FLOAT,
				scalar_count: 1024,
				sorted_values: (0..1024).map(|i| i as f32).collect(),
				entropy: 0.0,
				counts: None,
			},
			EligibleTensor {
				name: "b".into(),
				dims: vec![2048],
				data_type: DT_FLOAT16,
				scalar_count: 2048,
				sorted_values: (0..2048).map(|i| i as f32).collect(),
				entropy: 0.0,
				counts: None,
			},
		];

		let pools_a = build_pools(&donors_a);
		let pools_b = build_pools(&donors_a); // identical data

		let mut sel_a = DonorSelector::new(&key);
		let mut sel_b = DonorSelector::new(&key);
		let mut pa = pools_a;
		let mut pb = pools_b;

		let d1 = select_donor_natural(&mut sel_a, &mut pa);
		let d2 = select_donor_natural(&mut sel_b, &mut pb);
		assert_eq!(d1.name, d2.name, "Natural Ratio must be deterministic");
	}

	#[test]
	fn test_inject_extract_roundtrip_fp32() {
		let vals: Vec<f32> = (0..2048).map(|i| i as f32 * 0.001).collect();
		let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
		let tensor = crate::proto::TensorProto {
			name: Some("weight".into()),
			dims: vec![2048],
			data_type: Some(DT_FLOAT),
			raw_data: Some(raw),
			..Default::default()
		};
		let model = crate::proto::ModelProto {
			graph: Some(crate::proto::GraphProto {
				initializer: vec![tensor],
				..Default::default()
			}),
			..Default::default()
		};

		let dir = std::env::temp_dir();
		let donor_path = dir.join("stnx_test_donor.onnx");
		let stego_path = dir.join("stnx_test_stego.onnx");
		let _ = std::fs::remove_file(&donor_path);
		let _ = std::fs::remove_file(&stego_path);

		helpers::save_model(&model, &donor_path).unwrap();

		let payload = b"Hello, stego world!";
		let passphrase = "test-passphrase-123";

		let mut success = false;
		for _ in 0..100 {
			let stream = crate::crypto::encrypt(payload, "test.txt", passphrase, 3).unwrap();
			if inject(&donor_path, &stream, passphrase, &stego_path, 0.70).is_ok() {
				success = true;
				break;
			}
		}
		assert!(success, "injection failed after 100 attempts");

		let (recovered_data, recovered_name) = extract(&stego_path, passphrase).unwrap();
		assert_eq!(
			recovered_data,
			payload.to_vec(),
			"roundtrip must recover payload"
		);
		assert_eq!(recovered_name, "test.txt");

		let _ = std::fs::remove_file(&donor_path);
		let _ = std::fs::remove_file(&stego_path);
	}
}
