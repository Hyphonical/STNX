//! Cryptographic foundation for Stnx.
//!
//! Implements the Argon2id-based key derivation, HMAC-SHA256 subkey derivation,
//! AES-256-GCM encryption/decryption, zstd compression, and payload stream framing
//! as specified in PLAN.md Sections 5 and 6.

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Nonce};
use argon2::Argon2;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can arise from cryptographic operations.
#[derive(Debug)]
pub enum CryptoError {
	/// AES-GCM decryption failed (wrong passphrase or tampered data).
	Decryption,
	/// Zstd compression failed.
	Compress(String),
	/// Zstd decompression failed.
	Decompress(String),
	/// The filename in the header exceeded the 127-byte limit.
	FilenameTooLong,
	/// The header data was malformed (wrong length).
	InvalidHeader,
	/// The decrypted stream failed SHA-256 integrity check.
	IntegrityMismatch,
}

impl std::fmt::Display for CryptoError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Decryption => write!(
				f,
				"AES-GCM decryption failed (wrong passphrase or tampered data)"
			),
			Self::Compress(e) => write!(f, "zstd compression error: {e}"),
			Self::Decompress(e) => write!(f, "zstd decompression error: {e}"),
			Self::FilenameTooLong => write!(f, "filename exceeds 127-byte limit"),
			Self::InvalidHeader => write!(f, "malformed header data"),
			Self::IntegrityMismatch => write!(f, "SHA-256 integrity check failed"),
		}
	}
}

impl std::error::Error for CryptoError {}

// ---------------------------------------------------------------------------
// Constants  (PLAN.md Sections 5 & 6)
// ---------------------------------------------------------------------------

/// Domain-separation label for the salt derivation hash.
const KDF_SALT_LABEL: &[u8] = b"stnx.kdf.v1";

/// Length of the KDF salt in bytes.
const SALT_LEN: usize = 16;

/// Argon2id memory cost (64 MiB).
const ARGON2_MEMORY: u32 = 65_536;

/// Argon2id time cost.
const ARGON2_ITERATIONS: u32 = 3;

/// Argon2id parallelism (lanes).
const ARGON2_PARALLELISM: u32 = 4;

/// Master secret length (256 bits).
const MASTER_SECRET_LEN: usize = 32;

/// Subkey labels (PLAN.md Section 6).
const LABEL_ENC: &[u8] = b"stnx.enc";
const LABEL_NAME: &[u8] = b"stnx.name";
const LABEL_PROFILE: &[u8] = b"stnx.profile";
const LABEL_PAD: &[u8] = b"stnx.pad";

/// AES-256-GCM nonce length (96 bits).
const NONCE_LEN: usize = 12;

/// GCM authentication tag length (128 bits).
const TAG_LEN: usize = 16;

/// Payload header length (PLAN.md Section 5).
const HEADER_LEN: usize = 172;
/// Offset within header for format version (4 B).
const HEADER_VERSION_OFF: usize = 0;
/// Offset within header for uncompressed file length (8 B LE).
const HEADER_UNCOMP_LEN_OFF: usize = 4;
/// Offset within header for SHA-256 of raw file (32 B).
const HEADER_SHA256_OFF: usize = 12;
/// Offset within header for null-terminated filename (128 B).
const HEADER_FILENAME_OFF: usize = 44;
/// Maximum filename length (127 bytes + null terminator).
const FILENAME_MAX: usize = 127;

/// Format version for the payload stream.
const FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Derived secrets
// ---------------------------------------------------------------------------

/// The four subkeys derived from the master secret (PLAN.md Section 6).
#[derive(Debug)]
pub struct Subkeys {
	/// AES-256-GCM encryption / decryption key.
	pub enc: [u8; 32],
	/// CSPRNG seed for generating stego tensor names.
	pub name: [u8; 32],
	/// CSPRNG seed for selecting the donor sequence.
	pub profile: [u8; 32],
	/// CSPRNG seed for generating synthetic padding bytes.
	pub pad: [u8; 32],
}

/// Derive the deterministic Argon2id salt from a passphrase.
///
/// `salt = SHA-256("stnx.kdf.v1" || passphrase)[:16]`
fn derive_salt(passphrase: &str) -> [u8; SALT_LEN] {
	let mut hasher = Sha256::new();
	hasher.update(KDF_SALT_LABEL);
	hasher.update(passphrase.as_bytes());
	let hash = hasher.finalize();
	let mut salt = [0u8; SALT_LEN];
	salt.copy_from_slice(&hash[..SALT_LEN]);
	salt
}

/// Derive the master secret `K` from a passphrase via Argon2id.
fn derive_master_secret(passphrase: &str) -> Result<[u8; MASTER_SECRET_LEN], CryptoError> {
	let salt = derive_salt(passphrase);
	let argon2 = Argon2::new(
		argon2::Algorithm::Argon2id,
		argon2::Version::V0x13,
		argon2::Params::new(
			ARGON2_MEMORY,
			ARGON2_ITERATIONS,
			ARGON2_PARALLELISM,
			Some(MASTER_SECRET_LEN),
		)
		.unwrap(),
	);
	let mut secret = [0u8; MASTER_SECRET_LEN];
	argon2
		.hash_password_into(passphrase.as_bytes(), &salt, &mut secret)
		.map_err(|e| CryptoError::Compress(e.to_string()))?;
	Ok(secret)
}

/// Derive a 32-byte subkey via `HMAC-SHA256(K, label)`.
///
/// Implemented directly with SHA-256 to avoid version conflicts between
/// the `hmac` and `aes-gcm` crate families on `crypto_common`.
fn derive_subkey(master: &[u8; 32], label: &[u8]) -> Result<[u8; 32], CryptoError> {
	// HMAC-SHA256(K, label) = SHA256((K ^ opad) || SHA256((K ^ ipad) || label))

	let mut k_ipad = [0x36u8; 64];
	let mut k_opad = [0x5Cu8; 64];
	for i in 0..master.len() {
		k_ipad[i] ^= master[i];
		k_opad[i] ^= master[i];
	}

	let inner = Sha256::new()
		.chain_update(k_ipad)
		.chain_update(label)
		.finalize();

	let result = Sha256::new()
		.chain_update(k_opad)
		.chain_update(inner)
		.finalize();

	let mut key = [0u8; 32];
	key.copy_from_slice(&result);
	Ok(key)
}

/// Derive the full set of four subkeys from a passphrase.
pub fn derive_subkeys(passphrase: &str) -> Result<Subkeys, CryptoError> {
	let master = derive_master_secret(passphrase)?;
	Ok(Subkeys {
		enc: derive_subkey(&master, LABEL_ENC)?,
		name: derive_subkey(&master, LABEL_NAME)?,
		profile: derive_subkey(&master, LABEL_PROFILE)?,
		pad: derive_subkey(&master, LABEL_PAD)?,
	})
}

// ---------------------------------------------------------------------------
// Payload stream framing  (PLAN.md Section 5)
// ---------------------------------------------------------------------------

/// Build the 172-byte plaintext header.
///
/// Fields:
/// -  4 B: format version (u32 LE)
/// -  8 B: uncompressed file length (u64 LE)
/// - 32 B: SHA-256 of raw file
/// - 128 B: null-terminated filename, remainder zeroed
fn build_header(
	raw_file: &[u8],
	uncompressed_len: u64,
	filename: &str,
) -> Result<[u8; HEADER_LEN], CryptoError> {
	if filename.len() > FILENAME_MAX {
		return Err(CryptoError::FilenameTooLong);
	}

	let file_hash = Sha256::digest(raw_file);

	let mut header = [0u8; HEADER_LEN];

	header[HEADER_VERSION_OFF..HEADER_VERSION_OFF + 4]
		.copy_from_slice(&FORMAT_VERSION.to_le_bytes());
	header[HEADER_UNCOMP_LEN_OFF..HEADER_UNCOMP_LEN_OFF + 8]
		.copy_from_slice(&uncompressed_len.to_le_bytes());
	header[HEADER_SHA256_OFF..HEADER_SHA256_OFF + 32].copy_from_slice(&file_hash);

	let name_bytes = filename.as_bytes();
	let name_len = name_bytes.len().min(FILENAME_MAX);
	header[HEADER_FILENAME_OFF..HEADER_FILENAME_OFF + name_len]
		.copy_from_slice(&name_bytes[..name_len]);
	// The byte at HEADER_FILENAME_OFF + name_len is already zero from initialisation.

	Ok(header)
}

/// Parse a 172-byte header back into its components.
pub struct ParsedHeader {
	pub format_version: u32,
	pub uncompressed_len: u64,
	pub file_hash: [u8; 32],
	pub filename: String,
}

fn parse_header(data: &[u8; HEADER_LEN]) -> Result<ParsedHeader, CryptoError> {
	let format_version = u32::from_le_bytes(
		data[HEADER_VERSION_OFF..HEADER_VERSION_OFF + 4]
			.try_into()
			.map_err(|_| CryptoError::InvalidHeader)?,
	);
	let uncompressed_len = u64::from_le_bytes(
		data[HEADER_UNCOMP_LEN_OFF..HEADER_UNCOMP_LEN_OFF + 8]
			.try_into()
			.map_err(|_| CryptoError::InvalidHeader)?,
	);

	let mut file_hash = [0u8; 32];
	file_hash.copy_from_slice(&data[HEADER_SHA256_OFF..HEADER_SHA256_OFF + 32]);

	// Find null terminator within the 128-byte filename field.
	let name_start = HEADER_FILENAME_OFF;
	let name_end = data[name_start..]
		.iter()
		.position(|&b| b == 0)
		.unwrap_or(FILENAME_MAX);
	let filename = String::from_utf8(data[name_start..name_start + name_end].to_vec())
		.map_err(|_| CryptoError::InvalidHeader)?;

	Ok(ParsedHeader {
		format_version,
		uncompressed_len,
		file_hash,
		filename,
	})
}

/// Encrypt payload: compress → prepend header → AES-256-GCM.
///
/// Returns the assembled stream:
/// `[12 B nonce] [8 B ciphertext+tag length LE] [N B ciphertext] [16 B GCM tag]`
pub fn encrypt_payload(
	raw_file: &[u8],
	filename: &str,
	enc_key: &[u8; 32],
	zstd_level: i32,
) -> Result<Vec<u8>, CryptoError> {
	// Compress
	let compressed = zstd::encode_all(std::io::Cursor::new(raw_file), zstd_level)
		.map_err(|e| CryptoError::Compress(e.to_string()))?;

	// Build header
	let uncompressed_len = raw_file.len() as u64;
	let header = build_header(raw_file, uncompressed_len, filename)?;

	// Plaintext = header || compressed
	let mut plaintext = header.to_vec();
	plaintext.extend_from_slice(&compressed);

	// Encrypt with AES-256-GCM
	let cipher = Aes256Gcm::new_from_slice(enc_key).expect("valid 32-byte key");
	let nonce_vec = Aes256Gcm::generate_nonce(&mut OsRng);
	let nonce_bytes: [u8; NONCE_LEN] = nonce_vec.into();

	let ciphertext = cipher
		.encrypt(Nonce::from_slice(&nonce_bytes), plaintext.as_ref())
		.map_err(|_| CryptoError::Decryption)?;

	// Assemble stream: nonce(12) || length(8) || ciphertext+tag(N)
	let stream_len = ciphertext.len() as u64;
	let mut stream = Vec::with_capacity(NONCE_LEN + 8 + ciphertext.len());
	stream.extend_from_slice(&nonce_bytes);
	stream.extend_from_slice(&stream_len.to_le_bytes());
	stream.extend_from_slice(&ciphertext);

	Ok(stream)
}

/// Decrypt payload: AES-256-GCM → parse header → verify SHA-256 → decompress.
///
/// Input is the assembled stream produced by `encrypt_payload`.
pub fn decrypt_payload(
	stream: &[u8],
	enc_key: &[u8; 32],
) -> Result<(Vec<u8>, String), CryptoError> {
	if stream.len() < NONCE_LEN + 8 + TAG_LEN {
		return Err(CryptoError::Decryption);
	}

	// Parse nonce
	let nonce = Nonce::from_slice(&stream[..NONCE_LEN]);

	// Parse stream length
	let stream_len = u64::from_le_bytes(
		stream[NONCE_LEN..NONCE_LEN + 8]
			.try_into()
			.map_err(|_| CryptoError::Decryption)?,
	) as usize;

	// Ciphertext starts after nonce + length
	let ct_start = NONCE_LEN + 8;
	if ct_start + stream_len > stream.len() {
		return Err(CryptoError::Decryption);
	}
	let ciphertext = &stream[ct_start..ct_start + stream_len];

	// Decrypt
	let cipher = Aes256Gcm::new_from_slice(enc_key).expect("valid 32-byte key");
	let plaintext = cipher
		.decrypt(nonce, ciphertext)
		.map_err(|_| CryptoError::Decryption)?;

	// Split into header + compressed remainder
	if plaintext.len() < HEADER_LEN {
		return Err(CryptoError::Decryption);
	}
	let mut header_arr = [0u8; HEADER_LEN];
	header_arr.copy_from_slice(&plaintext[..HEADER_LEN]);
	let header = parse_header(&header_arr)?;
	let compressed = &plaintext[HEADER_LEN..];

	// Decompress
	let decompressed: Vec<u8> = zstd::decode_all(std::io::Cursor::new(compressed))
		.map_err(|e| CryptoError::Decompress(e.to_string()))?;

	// Verify SHA-256
	let actual_hash = Sha256::digest(&decompressed);
	if actual_hash.as_slice() != header.file_hash {
		return Err(CryptoError::IntegrityMismatch);
	}

	Ok((decompressed, header.filename))
}

// ---------------------------------------------------------------------------
// Convenience: derive subkeys from passphrase string
// ---------------------------------------------------------------------------

/// Convenience wrapper: derive subkeys and encrypt a payload in one call.
pub fn encrypt(
	raw_file: &[u8],
	filename: &str,
	passphrase: &str,
	zstd_level: i32,
) -> Result<Vec<u8>, CryptoError> {
	let subkeys = derive_subkeys(passphrase)?;
	encrypt_payload(raw_file, filename, &subkeys.enc, zstd_level)
}

/// Convenience wrapper: decrypt a stream and recover the original file + filename.
pub fn decrypt(stream: &[u8], passphrase: &str) -> Result<(Vec<u8>, String), CryptoError> {
	let subkeys = derive_subkeys(passphrase)?;
	decrypt_payload(stream, &subkeys.enc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_salt_derivation() {
		let salt = derive_salt("hunter2");
		assert_eq!(salt.len(), 16, "salt must be 16 bytes");

		// Deterministic
		let salt2 = derive_salt("hunter2");
		assert_eq!(salt, salt2, "salt must be deterministic");

		// Different passphrase → different salt
		let salt3 = derive_salt("hunter3");
		assert_ne!(
			salt, salt3,
			"different passphrases must produce different salts"
		);
	}

	#[test]
	fn test_master_secret_derivation() {
		let secret = derive_master_secret("test-passphrase").expect("KDF should succeed");
		assert_eq!(secret.len(), 32, "master secret must be 32 bytes");

		// Deterministic
		let secret2 = derive_master_secret("test-passphrase").expect("KDF should succeed");
		assert_eq!(secret, secret2, "master secret must be deterministic");
	}

	#[test]
	fn test_subkey_derivation() {
		let subkeys = derive_subkeys("test-passphrase").expect("subkey derivation should succeed");
		assert_eq!(subkeys.enc.len(), 32);
		assert_eq!(subkeys.name.len(), 32);
		assert_eq!(subkeys.profile.len(), 32);
		assert_eq!(subkeys.pad.len(), 32);

		// All four subkeys are distinct
		let mut all = vec![subkeys.enc, subkeys.name, subkeys.profile, subkeys.pad];
		all.sort();
		all.dedup();
		assert_eq!(all.len(), 4, "all four subkeys must be distinct");
	}

	#[test]
	fn test_header_roundtrip() {
		let data = b"hello, world! this is a test file with some content.";
		let header = build_header(data, data.len() as u64, "test.txt").expect("header build");
		assert_eq!(header.len(), HEADER_LEN);

		let parsed = parse_header(&header).expect("header parse");
		assert_eq!(parsed.format_version, FORMAT_VERSION);
		assert_eq!(parsed.uncompressed_len, data.len() as u64);
		assert_eq!(parsed.filename, "test.txt");

		let expected_hash = Sha256::digest(data);
		assert_eq!(parsed.file_hash, expected_hash.as_slice());
	}

	#[test]
	fn test_filename_too_long() {
		let long_name = "a".repeat(200);
		let data = b"some data";
		let result = build_header(data, data.len() as u64, &long_name);
		assert!(
			result.is_err(),
			"filenames over 127 bytes should be rejected"
		);
	}

	#[test]
	fn test_encrypt_decrypt_roundtrip() {
		let passphrase = "correct-horse-battery-staple";
		let filename = "secret.jpg";
		let raw = b"this is the secret payload content that will be hidden inside an ONNX model.";

		let stream = encrypt(raw, filename, passphrase, 3).expect("encryption should succeed");
		assert!(
			stream.len() > NONCE_LEN + 8,
			"stream must contain nonce and length"
		);

		let (recovered, recovered_name) =
			decrypt(&stream, passphrase).expect("decryption should succeed");
		assert_eq!(recovered, raw, "decrypted payload must match original");
		assert_eq!(recovered_name, filename, "recovered filename must match");
	}

	#[test]
	fn test_wrong_passphrase_fails() {
		let raw = b"some secret data";
		let stream = encrypt(raw, "doc.pdf", "correct-passphrase", 3).expect("encryption");

		let result = decrypt(&stream, "wrong-passphrase");
		assert!(result.is_err(), "wrong passphrase must fail decryption");
	}

	#[test]
	fn test_tampered_stream_fails() {
		let raw = b"important data";
		let mut stream = encrypt(raw, "data.bin", "passphrase", 3).expect("encryption");

		// Corrupt a byte in the ciphertext region
		let corrupt_idx = stream.len() / 2;
		stream[corrupt_idx] ^= 0xFF;

		let result = decrypt(&stream, "passphrase");
		assert!(result.is_err(), "tampered stream must fail decryption");
	}

	#[test]
	fn test_empty_payload() {
		let raw = b"";
		let stream = encrypt(raw, "empty.bin", "p4ss", 3).expect("encryption of empty file");
		let (recovered, name) = decrypt(&stream, "p4ss").expect("decryption of empty file");
		assert_eq!(recovered, b"", "empty payload roundtrip");
		assert_eq!(name, "empty.bin");
	}

	#[test]
	fn test_large_payload() {
		let raw: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
		let stream = encrypt(&raw, "large.dat", "s3cr3t", 3).expect("encryption of large file");
		let (recovered, _) = decrypt(&stream, "s3cr3t").expect("decryption of large file");
		assert_eq!(recovered, raw, "large payload must roundtrip correctly");
	}
}
