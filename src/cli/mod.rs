use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Constellation Encoding for ONNX Steganography.
#[derive(Parser, Debug)]
#[command(name = "stnx", author, version, about, long_about = None)]
pub struct Cli {
	#[command(subcommand)]
	pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
	/// Profile a donor ONNX model and report capacity
	Profile {
		/// Path to the donor ONNX model
		model: PathBuf,
		/// Payload utilization factor (alpha), defaults to 0.70 (70%). Set to 1.0 or higher for unlimited.
		#[arg(short, long, default_value_t = 0.70)]
		alpha: f64,
	},
	/// Inject a payload into a donor ONNX model
	Inject {
		/// Path to the donor ONNX model
		model: PathBuf,
		/// Path to the payload file to hide
		payload: PathBuf,
		/// Passphrase for key derivation (also via STNX_PASSPHRASE env)
		#[arg(short, long, env = "STNX_PASSPHRASE")]
		passphrase: String,
		/// Zstd compression level (default: 3)
		#[arg(long, default_value_t = 3)]
		zstd_level: i32,
		/// Output path for the stego model
		#[arg(short, long)]
		out: Option<PathBuf>,
		/// Payload utilization factor (alpha), defaults to 0.70 (70%). Set to 1.0 or higher for unlimited.
		#[arg(short, long, default_value_t = 0.70)]
		alpha: f64,
	},
	/// Extract a hidden payload from a stego ONNX model
	Extract {
		/// Path to the stego ONNX model
		model: PathBuf,
		/// Passphrase for key derivation (also via STNX_PASSPHRASE env)
		#[arg(short, long, env = "STNX_PASSPHRASE")]
		passphrase: String,
		/// Output path for the recovered file
		#[arg(short, long)]
		out: Option<PathBuf>,
	},
	/// Verify a stego ONNX model's statistical integrity
	Verify {
		/// Path to the stego ONNX model
		model: PathBuf,
		/// Passphrase for key derivation (also via STNX_PASSPHRASE env)
		#[arg(short, long, env = "STNX_PASSPHRASE")]
		passphrase: String,
	},
}
