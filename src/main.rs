pub mod cli;
pub mod crypto;
pub mod proto;
pub mod stats;
pub mod stego;

use clap::Parser;
use cli::{Cli, Commands};
use owo_colors::OwoColorize;

fn main() {
	let cli = Cli::parse();

	match cli.command {
		Commands::Profile { model, alpha } => {
			eprintln!("{} Profiling {} …", "…".cyan(), model.display().cyan());
			match stego::profile(&model, alpha) {
				Ok(report) => {
					println!("{report}");
				}
				Err(e) => {
					eprintln!("{} {e}", "✗".red().bold());
				}
			}
		}
		Commands::Inject {
			model,
			payload,
			passphrase,
			zstd_level,
			out,
			alpha,
			dtype_bias,
		} => {
			// Read payload file
			let payload_bytes = match std::fs::read(&payload) {
				Ok(b) => b,
				Err(e) => {
					eprintln!(
						"{} Failed to read payload '{}': {e}",
						"✗".red().bold(),
						payload.display().cyan()
					);
					return;
				}
			};

			let filename = payload
				.file_name()
				.and_then(|s| s.to_str())
				.unwrap_or("payload");

			let out_path = out.unwrap_or_else(|| model.with_extension("stego.onnx"));
			let mut success = false;

			for attempt in 1..=50 {
				// Encrypt payload → assembled stream
				let stream =
					match crypto::encrypt(&payload_bytes, filename, &passphrase, zstd_level) {
						Ok(s) => s,
						Err(e) => {
							eprintln!("{} Encryption failed: {e}", "✗".red().bold());
							return;
						}
					};

				if attempt == 1 {
					eprintln!(
						"{} {} B → {} B encrypted stream",
						"…".cyan(),
						payload_bytes.len().to_string().white(),
						stream.len().to_string().white()
					);
					eprintln!("{} Injecting into {} …", "…".cyan(), model.display().cyan());
				}

				match stego::inject(&model, &stream, &passphrase, &out_path, alpha, dtype_bias) {
					Ok(()) => {
						println!(
							"{} {}  ({} B payload → {} B stream){}",
							"✓".green().bold(),
							out_path.display().cyan(),
							payload_bytes.len().to_string().white(),
							stream.len().to_string().white(),
							if attempt > 1 {
								format!(" (took {} attempts)", attempt)
							} else {
								"".to_string()
							}
						);
						success = true;
						break;
					}
					Err(e) => {
						if attempt == 50 {
							eprintln!(
								"{} Injection failed after 50 attempts: {e}",
								"✗".red().bold()
							);
						}
					}
				}
			}

			if !success {
				std::process::exit(1);
			}
		}
		Commands::Extract {
			model,
			passphrase,
			out,
			dtype_bias,
		} => {
			eprintln!(
				"{} Extracting from {} …",
				"…".cyan(),
				model.display().cyan()
			);

			match stego::extract(&model, &passphrase, dtype_bias) {
				Ok((data, filename)) => {
					let out_name = if filename.is_empty() {
						"recovered_payload".to_string()
					} else {
						filename
					};
					let out_path = out.unwrap_or_else(|| {
						model
							.parent()
							.unwrap_or_else(|| std::path::Path::new("."))
							.join(&out_name)
					});
					if let Err(e) = std::fs::write(&out_path, &data) {
						eprintln!(
							"{} Failed to write recovered file '{}': {e}",
							"✗".red().bold(),
							out_path.display().cyan()
						);
					} else {
						println!(
							"{} Recovered {} bytes → {}",
							"✓".green().bold(),
							data.len().to_string().white(),
							out_path.display().cyan()
						);
					}
				}
				Err(e) => {
					eprintln!("{} Extraction failed: {e}", "✗".red().bold());
				}
			}
		}
		Commands::Verify {
			model,
			passphrase,
			dtype_bias,
		} => {
			eprintln!("{} Verifying {} …", "…".cyan(), model.display().cyan());

			match stego::verify(&model, &passphrase, dtype_bias) {
				Ok(report) => {
					if report.all_pass {
						println!(
							"{} All {} chunks passed statistical verification",
							"✓".green().bold(),
							report.total_chunks.to_string().white()
						);
					} else {
						eprintln!(
							"{} {} / {} chunks FAILED statistical verification",
							"✗".red().bold(),
							report.failed_chunks.to_string().white(),
							report.total_chunks.to_string().white()
						);
					}

					for chunk in &report.chunks {
						let status = if chunk.ks_pass && chunk.chi2_pass {
							"✓".green().to_string()
						} else {
							"✗".red().to_string()
						};
						println!(
							"  {} {}  (donor: {})",
							status,
							chunk.stego_name.cyan(),
							chunk.donor_name.cyan(),
						);

						let ks_mark = if chunk.ks_pass {
							"✓".green().to_string()
						} else {
							"✗".red().to_string()
						};
						println!(
							"      K–S   {}  D={:.6}  (crit={:.6})",
							ks_mark, chunk.ks_stat, chunk.ks_crit
						);

						let chi2_mark = if chunk.chi2_pass {
							"✓".green().to_string()
						} else {
							"✗".red().to_string()
						};
						println!(
							"      χ²    {}  χ²={:.2}  (crit={:.2})",
							chi2_mark, chunk.chi2_stat, chunk.chi2_crit
						);
					}
				}
				Err(e) => {
					eprintln!("{} Verification failed: {e}", "✗".red().bold());
				}
			}
		}
	}
}
