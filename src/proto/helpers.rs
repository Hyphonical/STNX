//! ONNX model loading, profiling, and TensorProto construction helpers.
//!
//! Provides the bridge between the protobuf representation and the stego
//! engine: load/save models, enumerate eligible initializers (FP32 / FP16),
//! extract sorted scalar values, build new TensorProto entries.

use crate::proto;
use prost::Message;
use std::path::Path;

/// Re-export common proto types used throughout helpers.
pub(crate) type TensorProto = proto::TensorProto;
pub(crate) type ModelProto = proto::ModelProto;

// ---------------------------------------------------------------------------
// Data-type constants  (from onnx::tensor_proto::DataType)
// ---------------------------------------------------------------------------

/// ONNX `TensorProto.DataType` value for `FLOAT` (IEEE 754 single-precision).
pub(crate) const DT_FLOAT: i32 = 1;

/// ONNX `TensorProto.DataType` value for `FLOAT16` (IEEE 754 half-precision).
pub(crate) const DT_FLOAT16: i32 = 10;

/// ONNX `TensorProto.DataType` value for `UINT8`.
pub(crate) const DT_UINT8: i32 = 2;

/// ONNX `TensorProto.DataType` value for `INT8`.
pub(crate) const DT_INT8: i32 = 3;

/// Minimum number of scalar elements a donor tensor must have to be eligible.
const MIN_ELEMENTS: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from ONNX model operations.
#[derive(Debug)]
pub enum ProtoError {
	/// Failed to read the file from disk.
	Io(std::io::Error),
	/// Failed to decode the protobuf (invalid or corrupt ONNX).
	Decode(prost::DecodeError),
	/// Failed to encode the protobuf.
	Encode(prost::EncodeError),
	/// The model has no graph.
	NoGraph,
	/// The model has no eligible FP32 or FP16 tensors at all.
	NoEligibleTensors,
	/// The raw_data field is missing on a tensor that should have it.
	MissingRawData(String),
	/// The raw_data length is not a multiple of the element byte width.
	MisalignedRawData {
		name: String,
		len: usize,
		elem_size: usize,
	},
}

impl std::fmt::Display for ProtoError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Io(e) => write!(f, "I/O error: {e}"),
			Self::Decode(e) => write!(f, "protobuf decode error: {e}"),
			Self::Encode(e) => write!(f, "protobuf encode error: {e}"),
			Self::NoGraph => write!(f, "model has no graph"),
			Self::NoEligibleTensors => write!(f, "no eligible FP32 or FP16 tensors found"),
			Self::MissingRawData(name) => {
				write!(f, "tensor '{name}' has no raw_data field")
			}
			Self::MisalignedRawData {
				name,
				len,
				elem_size,
			} => {
				write!(
					f,
					"tensor '{name}' raw_data length {len} is not a multiple of element size {elem_size}"
				)
			}
		}
	}
}

impl std::error::Error for ProtoError {}

impl From<std::io::Error> for ProtoError {
	fn from(e: std::io::Error) -> Self {
		Self::Io(e)
	}
}

impl From<prost::DecodeError> for ProtoError {
	fn from(e: prost::DecodeError) -> Self {
		Self::Decode(e)
	}
}

impl From<prost::EncodeError> for ProtoError {
	fn from(e: prost::EncodeError) -> Self {
		Self::Encode(e)
	}
}

// ---------------------------------------------------------------------------
// Load / Save
// ---------------------------------------------------------------------------

/// Load an ONNX model from a file path.
pub fn load_model(path: &Path) -> Result<ModelProto, ProtoError> {
	let bytes = std::fs::read(path)?;
	let model = ModelProto::decode(bytes.as_slice())?;
	Ok(model)
}

/// Save an ONNX model to a file path.
pub fn save_model(model: &ModelProto, path: &Path) -> Result<(), ProtoError> {
	let bytes = model.encode_to_vec();
	Ok(std::fs::write(path, bytes)?)
}

// ---------------------------------------------------------------------------
// Eligible tensor descriptor  (PLAN.md §7.1)
// ---------------------------------------------------------------------------

/// A donor tensor that has passed the eligibility gates.
#[derive(Clone, Debug)]
pub struct EligibleTensor {
	/// Original name from the ONNX model.
	pub name: String,
	/// Shape dimensions.
	pub dims: Vec<i64>,
	/// ONNX data type (`1` = FLOAT, `10` = FLOAT16, `2` = UINT8, `3` = INT8).
	pub data_type: i32,
	/// Total number of scalar elements (= product of dims).
	pub scalar_count: usize,
	/// Sorted scalar values (FP16 widened to f32). Empty for INT8/UINT8.
	pub sorted_values: Vec<f32>,
	/// Empirical entropy (for INT8/UINT8), 0.0 for floats.
	pub entropy: f64,
	/// Histogram counts (for INT8/UINT8), None for floats.
	pub counts: Option<[usize; 256]>,
}

// ---------------------------------------------------------------------------
// FP16 → f32 widening  (PLAN.md §4.2.2, footnote on FP16 handling)
// ---------------------------------------------------------------------------

/// Widen an IEEE 754 half-precision (binary16) value to f32.
pub(crate) fn f16_to_f32(bit_pattern: u16) -> f32 {
	let sign = ((bit_pattern >> 15) & 0x1) as i32;
	let exp = ((bit_pattern >> 10) & 0x1F) as i32;
	let mant = (bit_pattern & 0x3FF) as i32;

	if exp == 0 {
		// Subnormal or zero
		if mant == 0 {
			// Signed zero
			f32::from_bits(if sign == 0 { 0u32 } else { 0x8000_0000u32 })
		} else {
			// Subnormal: normalize by shifting mantissa
			let mut m = mant;
			let mut e = -1i32;
			while (m & 0x400) == 0 {
				m <<= 1;
				e -= 1;
			}
			let biased_e = (127 + e) as u32;
			let mant_bits = (mant as u32) << 13;
			let sign_bit = (sign as u32) << 31;
			f32::from_bits(sign_bit | (biased_e << 23) | mant_bits)
		}
	} else if exp == 31 {
		// Infinity or NaN
		let sign_bit = (sign as u32) << 31;
		let exp_bits = 0xFFu32 << 23;
		let mant_bits = (mant as u32) << 13;
		f32::from_bits(sign_bit | exp_bits | mant_bits)
	} else {
		// Normal
		let sign_bit = (sign as u32) << 31;
		let biased_e = (exp + 127 - 15) as u32;
		let mant_bits = (mant as u32) << 13;
		f32::from_bits(sign_bit | (biased_e << 23) | mant_bits)
	}
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
	let bits = value.to_bits();
	let sign = ((bits >> 31) & 0x1) as u16;
	let exp = ((bits >> 23) & 0xFF) as i32;
	let mant = bits & 0x7F_FFFF;

	if exp == 0 {
		// Zero / subnormal → flush to zero
		sign << 15
	} else if exp == 0xFF {
		// Inf / NaN
		let exp_bits = 0x1Fu16 << 10;
		let mant_bits = (mant >> 13) as u16;
		(sign << 15) | exp_bits | mant_bits
	} else {
		// Normal: re-bias exponent
		let new_exp = exp - 127 + 15;
		if new_exp >= 31 {
			// Overflow to infinity
			(sign << 15) | (0x1Fu16 << 10)
		} else if new_exp <= 0 {
			// Underflow to zero
			sign << 15
		} else {
			let exp_bits = (new_exp as u16) << 10;
			let mant_bits = (mant >> 13) as u16;
			// Round-to-nearest-even by checking the dropped bits
			let dropped = mant & 0x1FFF;
			let round = if dropped > 0x1000 || (dropped == 0x1000 && (mant_bits & 1) == 1) {
				1u16
			} else {
				0
			};
			(sign << 15) | exp_bits | (mant_bits + round)
		}
	}
}

// ---------------------------------------------------------------------------
// Extract mapped bytes (indices) from TensorProto for INT8/UINT8
// ---------------------------------------------------------------------------

pub(crate) fn extract_indices(tensor: &TensorProto) -> Result<Vec<u8>, ProtoError> {
	let dt = tensor.data_type.unwrap_or(0);
	let raw = tensor.raw_data.as_ref().ok_or_else(|| {
		ProtoError::MissingRawData(tensor.name.as_deref().unwrap_or("<unnamed>").to_string())
	})?;

	match dt {
		DT_UINT8 => Ok(raw.clone()),
		DT_INT8 => {
			let indices = raw
				.iter()
				.map(|&b| ((b as i8) as i16 + 128) as u8)
				.collect();
			Ok(indices)
		}
		_ => Err(ProtoError::MissingRawData(
			tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
		)),
	}
}

// ---------------------------------------------------------------------------
// Extract scalars from TensorProto  (PLAN.md §7.1)
// ---------------------------------------------------------------------------

pub fn extract_scalars(tensor: &TensorProto) -> Result<Vec<f32>, ProtoError> {
	let dt = tensor.data_type.unwrap_or(0);
	let raw = tensor.raw_data.as_ref().ok_or_else(|| {
		ProtoError::MissingRawData(tensor.name.as_deref().unwrap_or("<unnamed>").to_string())
	})?;

	match dt {
		DT_FLOAT => {
			if raw.len() % 4 != 0 {
				return Err(ProtoError::MisalignedRawData {
					name: tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
					len: raw.len(),
					elem_size: 4,
				});
			}
			let scalars: Vec<f32> = raw
				.chunks_exact(4)
				.map(|chunk: &[u8]| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
				.collect();
			Ok(scalars)
		}
		DT_FLOAT16 => {
			if raw.len() % 2 != 0 {
				return Err(ProtoError::MisalignedRawData {
					name: tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
					len: raw.len(),
					elem_size: 2,
				});
			}
			let scalars: Vec<f32> = raw
				.chunks_exact(2)
				.map(|chunk: &[u8]| {
					let bits = u16::from_le_bytes(chunk.try_into().expect("2-byte chunk"));
					f16_to_f32(bits)
				})
				.collect();
			Ok(scalars)
		}
		_ => Err(ProtoError::MissingRawData(
			tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
		)),
	}
}

// ---------------------------------------------------------------------------
// Eligibility filtering  (PLAN.md §7.1, §11.2)
// ---------------------------------------------------------------------------

pub fn is_tensor_eligible(tensor: &TensorProto) -> bool {
	let dt = tensor.data_type.unwrap_or(0);

	// Quick element count from dims
	let scalar_count: usize = tensor.dims.iter().map(|&d| d as usize).product();
	if scalar_count < MIN_ELEMENTS {
		return false;
	}

	match dt {
		DT_FLOAT | DT_FLOAT16 => {
			// Try extracting scalars; if it fails, ineligible
			let values = match extract_scalars(tensor) {
				Ok(v) => v,
				Err(_) => return false,
			};

			// Check distinct values
			if values.len() < 256 {
				return false;
			}

			// Quick distinct-value check: sort, count distinct in one pass
			let mut sorted = values.to_vec();
			sorted.sort_unstable_by(|a, b| a.total_cmp(b));
			let distinct = sorted.windows(2).filter(|w| w[0] != w[1]).count() + 1;
			distinct >= 256
		}
		DT_INT8 | DT_UINT8 => {
			// INT8/UINT8 eligibility (INT8.md §5.1)
			let indices = match extract_indices(tensor) {
				Ok(v) => v,
				Err(_) => return false,
			};

			let mut counts = [0usize; 256];
			for &idx in &indices {
				counts[idx as usize] += 1;
			}

			// Minimum entropy check
			let entropy = crate::stats::empirical_entropy(&counts, scalar_count);
			entropy >= 4.0
		}
		_ => false,
	}
}

pub fn eligible_initializers(model: &ModelProto) -> Vec<EligibleTensor> {
	let graph = match model.graph.as_ref() {
		Some(g) => g,
		None => return vec![],
	};

	let mut results = Vec::new();

	for tensor in &graph.initializer {
		let dt = tensor.data_type.unwrap_or(0);
		let scalar_count: usize = tensor.dims.iter().map(|&d| d as usize).product();
		if scalar_count < MIN_ELEMENTS {
			continue;
		}

		match dt {
			DT_FLOAT | DT_FLOAT16 => {
				let values = match extract_scalars(tensor) {
					Ok(v) => v,
					Err(_) => continue,
				};

				// Sort and check distinct count
				let mut sorted = values;
				sorted.sort_unstable_by(|a, b| a.total_cmp(b));
				// Quick distinct count
				let distinct = sorted.windows(2).filter(|w| w[0] != w[1]).count() + 1;
				if distinct < 256 {
					continue;
				}

				results.push(EligibleTensor {
					name: tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
					dims: tensor.dims.clone(),
					data_type: dt,
					scalar_count,
					sorted_values: sorted,
					entropy: 0.0,
					counts: None,
				});
			}
			DT_INT8 | DT_UINT8 => {
				let indices = match extract_indices(tensor) {
					Ok(v) => v,
					Err(_) => continue,
				};

				let mut counts = [0usize; 256];
				for &idx in &indices {
					counts[idx as usize] += 1;
				}

				let entropy = crate::stats::empirical_entropy(&counts, scalar_count);
				if entropy < 4.0 {
					continue;
				}

				results.push(EligibleTensor {
					name: tensor.name.as_deref().unwrap_or("<unnamed>").to_string(),
					dims: tensor.dims.clone(),
					data_type: dt,
					scalar_count,
					sorted_values: Vec::new(),
					entropy,
					counts: Some(counts),
				});
			}
			_ => continue,
		}
	}

	// Sort by element count descending for Natural Ratio (INT8.md §3.2)
	results.sort_by(|a, b| b.scalar_count.cmp(&a.scalar_count));
	results
}

// ---------------------------------------------------------------------------
// Capacity computation  (PLAN.md §9.1)
// ---------------------------------------------------------------------------

/// Compute the total eligible element count across all eligible tensors.
pub fn total_eligible_elements(eligible: &[EligibleTensor]) -> usize {
	eligible.iter().map(|e| e.scalar_count).sum()
}

/// Compute the maximum payload size in bytes (PLAN.md §9.1).
///
/// $$C_{\text{max}} = \lfloor \alpha N \rfloor$$
pub fn max_payload_bytes(total_eligible_elements: usize, alpha: f64) -> usize {
	(alpha * total_eligible_elements as f64).floor() as usize
}

/// Compute the disk space overhead multiplier for eligible tensors.
///
/// Returns `(fp16_el, fp32_el, int8_el, uint8_el, avg_dtype_overhead)`.
pub fn dtype_overhead(eligible: &[EligibleTensor]) -> (usize, usize, usize, usize, f64) {
	let fp16: usize = eligible
		.iter()
		.filter(|e| e.data_type == DT_FLOAT16)
		.map(|e| e.scalar_count)
		.sum();
	let fp32: usize = eligible
		.iter()
		.filter(|e| e.data_type == DT_FLOAT)
		.map(|e| e.scalar_count)
		.sum();
	let int8: usize = eligible
		.iter()
		.filter(|e| e.data_type == DT_INT8)
		.map(|e| e.scalar_count)
		.sum();
	let uint8: usize = eligible
		.iter()
		.filter(|e| e.data_type == DT_UINT8)
		.map(|e| e.scalar_count)
		.sum();
	let total = (fp16 + fp32 + int8 + uint8) as f64;
	let avg = if total > 0.0 {
		(2.0 * fp16 as f64 + 4.0 * fp32 as f64 + 1.0 * int8 as f64 + 1.0 * uint8 as f64) / total
	} else {
		0.0
	};
	(fp16, fp32, int8, uint8, avg)
}

// ---------------------------------------------------------------------------
// Build new TensorProto  (PLAN.md §7.2)
// ---------------------------------------------------------------------------

/// Build a new TensorProto from raw byte data.
///
/// `data_type` controls what gets written: FLOAT/FLOAT16 take f32 values,
/// INT8/UINT8 take raw bytes directly.
pub fn build_tensor_from_raw(
	name: &str,
	dims: &[i64],
	data_type: i32,
	raw_data: Vec<u8>,
) -> TensorProto {
	TensorProto {
		dims: dims.to_vec(),
		data_type: Some(data_type),
		name: Some(name.to_string()),
		raw_data: Some(raw_data),
		..Default::default()
	}
}

/// Build a new TensorProto from f32 values.
///
/// - `data_type`: `DT_FLOAT` (1) or `DT_FLOAT16` (10)
/// - For FP32, each f32 is written directly as 4 LE bytes into `raw_data`.
/// - For FP16, each f32 is narrowed via `f32_to_f16` and written as 2 LE bytes.
pub fn build_tensor(name: &str, dims: &[i64], data_type: i32, values: &[f32]) -> TensorProto {
	let raw_data = match data_type {
		DT_FLOAT => {
			let mut buf = Vec::with_capacity(values.len() * 4);
			for &v in values {
				buf.extend_from_slice(&v.to_le_bytes());
			}
			buf
		}
		DT_FLOAT16 => {
			let mut buf = Vec::with_capacity(values.len() * 2);
			for &v in values {
				buf.extend_from_slice(&f32_to_f16(v).to_le_bytes());
			}
			buf
		}
		_ => unreachable!("build_tensor called with unsupported data_type {data_type}"),
	};

	TensorProto {
		dims: dims.to_vec(),
		data_type: Some(data_type),
		name: Some(name.to_string()),
		raw_data: Some(raw_data),
		..Default::default()
	}
}

/// Collect a set of all existing initializer names in a model (for collision
/// detection during injection).
pub fn existing_initializer_names(model: &ModelProto) -> Vec<String> {
	model
		.graph
		.as_ref()
		.map(|g| {
			g.initializer
				.iter()
				.filter_map(|t| t.name.clone())
				.collect()
		})
		.unwrap_or_default()
}

/// Count distinct initializers by dtype.
pub fn count_by_dtype(model: &ModelProto) -> (usize, usize, usize, usize) {
	let graph = match model.graph.as_ref() {
		Some(g) => g,
		None => return (0, 0, 0, 0),
	};

	let mut fp32 = 0usize;
	let mut fp16 = 0usize;
	let mut int8 = 0usize;
	let mut uint8 = 0usize;

	for t in &graph.initializer {
		match t.data_type.unwrap_or(0) {
			DT_FLOAT => fp32 += 1,
			DT_FLOAT16 => fp16 += 1,
			DT_INT8 => int8 += 1,
			DT_UINT8 => uint8 += 1,
			_ => {}
		}
	}

	(fp32, fp16, int8, uint8)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::proto;

	// ── FP16 / f32 conversion ──────────────────────────────────────────────

	#[test]
	fn test_f16_to_f32_zero() {
		// Positive zero
		let result = f16_to_f32(0x0000);
		assert_eq!(result, 0.0f32);
		assert!(result.is_sign_positive());

		// Negative zero
		let result = f16_to_f32(0x8000);
		assert_eq!(result, -0.0f32);
		assert!(result.is_sign_negative());
	}

	#[test]
	fn test_f16_to_f32_one() {
		// 1.0 in FP16 = 0x3C00
		let result = f16_to_f32(0x3C00);
		assert!((result - 1.0).abs() < f32::EPSILON);
	}

	#[test]
	fn test_f16_to_f32_roundtrip() {
		let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 3.140625, -42.0];
		for &v in &values {
			let f16 = f32_to_f16(v);
			let back = f16_to_f32(f16);
			assert_eq!(v, back, "roundtrip failed for {v} -> 0x{f16:04X} -> {back}");
		}
	}

	#[test]
	fn test_f16_inf_nan() {
		// Infinity
		let inf_f16 = 0x7C00u16;
		let inf = f16_to_f32(inf_f16);
		assert!(inf.is_infinite());

		// NaN
		let nan_f16 = 0x7E00u16;
		let nan = f16_to_f32(nan_f16);
		assert!(nan.is_nan());
	}

	// ── build_tensor ───────────────────────────────────────────────────────

	#[test]
	fn test_build_tensor_fp32() {
		let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
		let tensor = build_tensor("test", &[2, 2], DT_FLOAT, &values);
		assert_eq!(tensor.name.as_deref(), Some("test"));
		assert_eq!(tensor.dims, vec![2, 2]);
		assert_eq!(tensor.data_type, Some(DT_FLOAT));

		let extracted = extract_scalars(&tensor).unwrap();
		assert_eq!(extracted, values);
	}

	#[test]
	fn test_build_tensor_fp16() {
		let values: Vec<f32> = vec![1.0, -2.0, 3.5, -0.5];
		let tensor = build_tensor("fp16_test", &[4], DT_FLOAT16, &values);
		assert_eq!(tensor.data_type, Some(DT_FLOAT16));

		let extracted = extract_scalars(&tensor).unwrap();
		for (a, b) in extracted.iter().zip(values.iter()) {
			assert!(
				(a - b).abs() < 0.001,
				"FP16 build/extract mismatch: {a} vs {b}"
			);
		}
	}

	// ── build & extract ────────────────────────────────────────────────────

	#[test]
	fn test_extract_scalars_from_raw_data() {
		let vals: Vec<f32> = vec![3.100, 2.700, 1.600, 0.577];
		let mut raw = Vec::with_capacity(vals.len() * 4);
		for v in &vals {
			raw.extend_from_slice(&v.to_le_bytes());
		}

		let tensor = proto::TensorProto {
			name: Some("test".into()),
			dims: vec![4],
			data_type: Some(DT_FLOAT),
			raw_data: Some(raw),
			..Default::default()
		};

		let extracted = extract_scalars(&tensor).unwrap();
		assert_eq!(extracted, vals);
	}

	#[test]
	fn test_extract_scalars_missing_raw_data_errors() {
		let tensor = proto::TensorProto {
			name: Some("no_raw".into()),
			dims: vec![4],
			data_type: Some(DT_FLOAT),
			raw_data: None,
			..Default::default()
		};
		assert!(extract_scalars(&tensor).is_err());
	}

	// ── eligibility ────────────────────────────────────────────────────────

	#[test]
	fn test_is_tensor_eligible_fp32() {
		// Build a valid FP32 tensor with 1024 elements, all distinct
		let vals: Vec<f32> = (0..1024).map(|i| i as f32).collect();
		let mut raw = Vec::with_capacity(vals.len() * 4);
		for v in &vals {
			raw.extend_from_slice(&v.to_le_bytes());
		}
		let tensor = proto::TensorProto {
			name: Some("eligible".into()),
			dims: vec![32, 32],
			data_type: Some(DT_FLOAT),
			raw_data: Some(raw),
			..Default::default()
		};
		assert!(is_tensor_eligible(&tensor));
	}

	#[test]
	fn test_is_tensor_eligible_too_few_elements() {
		let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
		let mut raw = Vec::new();
		for v in &vals {
			raw.extend_from_slice(&v.to_le_bytes());
		}
		let tensor = proto::TensorProto {
			name: Some("too_small".into()),
			dims: vec![10, 10],
			data_type: Some(DT_FLOAT),
			raw_data: Some(raw),
			..Default::default()
		};
		assert!(!is_tensor_eligible(&tensor));
	}

	#[test]
	fn test_is_tensor_eligible_wrong_dtype() {
		let tensor = proto::TensorProto {
			name: Some("int32".into()),
			dims: vec![1000],
			data_type: Some(6), // INT32
			..Default::default()
		};
		assert!(!is_tensor_eligible(&tensor));
	}

	#[test]
	fn test_is_tensor_eligible_constant_values() {
		// 1024 elements all the same value → only 1 distinct value
		let vals: Vec<f32> = vec![42.0; 1024];
		let mut raw = Vec::new();
		for v in &vals {
			raw.extend_from_slice(&v.to_le_bytes());
		}
		let tensor = proto::TensorProto {
			name: Some("constant".into()),
			dims: vec![1024],
			data_type: Some(DT_FLOAT),
			raw_data: Some(raw),
			..Default::default()
		};
		assert!(!is_tensor_eligible(&tensor));
	}

	// ── counted ────────────────────────────────────────────────────────────

	#[test]
	fn test_count_by_dtype_empty() {
		let model = proto::ModelProto::default();
		let (fp32, fp16, int8, uint8) = count_by_dtype(&model);
		assert_eq!(fp32, 0);
		assert_eq!(fp16, 0);
		assert_eq!(int8, 0);
		assert_eq!(uint8, 0);
	}

	#[test]
	fn test_build_and_select_donor_from_model() {
		// Build a model with one FP32 and one FP16 eligible tensor
		let fp32_vals: Vec<f32> = (0..1024).map(|i| i as f32).collect();
		let fp32_raw: Vec<u8> = fp32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
		let fp32_tensor = proto::TensorProto {
			name: Some("weight.fp32".into()),
			dims: vec![32, 32],
			data_type: Some(DT_FLOAT),
			raw_data: Some(fp32_raw),
			..Default::default()
		};

		let fp16_vals: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
		let fp16_raw: Vec<u8> = fp16_vals
			.iter()
			.flat_map(|&v| f32_to_f16(v).to_le_bytes())
			.collect();
		let fp16_tensor = proto::TensorProto {
			name: Some("weight.fp16".into()),
			dims: vec![32, 32],
			data_type: Some(DT_FLOAT16),
			raw_data: Some(fp16_raw),
			..Default::default()
		};

		let model = proto::ModelProto {
			graph: Some(proto::GraphProto {
				initializer: vec![fp32_tensor, fp16_tensor],
				..Default::default()
			}),
			..Default::default()
		};

		let eligible = eligible_initializers(&model);
		assert_eq!(eligible.len(), 2, "both tensors should be eligible");

		let (fp32, fp16, int8, uint8) = count_by_dtype(&model);
		assert_eq!(fp32, 1);
		assert_eq!(fp16, 1);
		assert_eq!(int8, 0);
		assert_eq!(uint8, 0);

		let total = total_eligible_elements(&eligible);
		assert_eq!(total, 2048);

		let max_payload = max_payload_bytes(total, 0.70);
		assert_eq!(max_payload, 1433); // floor(0.70 * 2048)
	}
}
