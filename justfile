# STNX build orchestration
# Install `just` via: cargo install just

# Use PowerShell on Windows, sh on Unix
set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]
set shell := ["bash", "-uc"]

# Default recipe - build everything
default: dev

# Verify formatting before build-related tasks
fmt:
    cargo fmt --all --check

# Development build with debug symbols and no optimizations, also run the linter
dev: fmt lint
    cargo build

# Release build with all optimizations
release: fmt lint
    cargo build --release --locked

# Clean all build artifacts
clean:
    cargo clean

# Run cargo check
check: fmt lint
    cargo check

# Run clippy linter
lint: fmt
    cargo clippy --all-targets --all-features -- -D warnings

# Run tests
test: fmt lint
    cargo test