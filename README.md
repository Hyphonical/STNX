# Constellation Encoding for ONNX Steganography

**Stnx** is a standalone steganographic system that embeds arbitrary files into existing ONNX models by appending statistically indistinguishable, unused weight tensors. The resulting model remains structurally valid, executes identically to the original, and leaks no external metadata about the hidden payload. This document specifies the complete architecture, mathematical foundations, operational workflow, and interface contract for the reference Rust implementation.

---

## 1. Project Synopsis

Stnx occupies a narrow but critical niche: it is a **capacity-bound, statistical steganography engine** for Protocol Buffer-serialized neural network checkpoints. It neither mutates existing graph topology nor alters weights referenced by inference paths. Instead, it injects payload data into the dead space of `GraphProto.initializer` as new, unreferenced `TensorProto` blobs.

The method is **Strategy B — Constellation Encoding**:

1. Compress and encrypt the payload with authenticated encryption.
2. Shard the ciphertext into numerous small chunks, with chunk sizes determined organically by the shape of whichever donor tensor is selected for each chunk.
3. Encode each chunk's bytes as IEEE 754 float values drawn directly from the donor tensor's own sorted population, such that the marginal distribution of the encoded chunk is statistically indistinguishable from the donor.
4. Instantiate the encoded values into unused initializers with plausible checkpoint-style names and shapes cloned from real architecture archetypes.

The result is a constellation of tensors: no single outlier, no entropy anomaly, no size anomaly, and zero impact on forward inference. Because chunks inherit both shape and dtype from their respective donors, a mixed-precision model produces stego tensors that are individually and collectively consistent with the surrounding weight population.

---

## 2. ONNX Structural Prerequisites

An ONNX file is a single `ModelProto` serialized via Protocol Buffers. Relevant fields:

| Field | Role |
|-------|------|
| `graph.input[]` | Declares execution-time inputs |
| `graph.output[]` | Declares execution-time outputs |
| `graph.node[]` | Operator topology (Conv, MatMul, etc.) |
| `graph.initializer[]` | **Static weight tensors** — dense blobs of raw bytes |

Each `TensorProto` in `initializer[]` carries:

- `name`: referenced by `node.input[]` (or not referenced at all)
- `dims[]`: shape dimensions
- `data_type`: scalar type enum (`1 = FLOAT`, `10 = FLOAT16`, etc.)
- `raw_data`: a flat byte string of tightly packed, little-endian scalars

**The crucial observation:** ONNX Runtime builds an execution plan by traversing `node[]` from outputs to inputs. Any `initializer` not transitively referenced by the active graph is carried faithfully inside the protobuf serialization. Appending unused initializers is therefore a no-op with respect to the model's semantic content, and the bytes are preserved exactly in the serialized file as long as nothing re-serializes the model through a runtime optimizer.

Stnx targets `FLOAT` (FP32) and `FLOAT16` (FP16) tensors as donors. These are the two dominant weight formats in production neural network checkpoints. Both are eligible, and both may coexist within the same donor pool for a single injection. Quantized formats — `INT4`, block-wise `QINT8`, `UINT8` lookup tables — are **incompatible** because their discrete, low-cardinality value spaces cannot absorb high-entropy payload bytes without producing statistically lethal histograms. The tool rejects such tensors at the profiling phase.

---

## 3. Threat Model and Design Goals

| Goal | Requirement |
|------|-------------|
| **Functional invariance** | The stego model must pass `onnx.checker` and produce **bitwise-identical** inference outputs versus the donor on the same inputs, if inference were performed. Existing weights are never touched. |
| **Statistical invisibility** | No per-tensor entropy test, K–S test, or byte-frequency chi-squared test may reject an injected tensor when compared against its donor archetype at α = 0.05. |
| **Metadata cleanliness** | No JSON sidecars, no `metadata_props` leakage, no plaintext filenames. All retrieval state must be reconstructible from the passphrase and the `.onnx` file alone. |
| **Authenticated confidentiality** | Payload must be confidentiality-protected and integrity-verified; extraction from a corrupted or tampered model must fail deterministically. |

Stnx does **not** attempt to defeat an analyst who possesses the exact original donor model and performs a direct tensor diff. That scenario is out of scope; the tool targets automated scanners and casual inspection, not targeted forensic comparison against a known-good reference checkpoint. The primary use case is long-term storage inside publicly distributed ONNX model files, where the stego file is never loaded through an optimizing runtime.

---

## 4. Constellation Encoding: The Core Method

### 4.1 Chunking and Plausibility

A single injected tensor large enough to carry 200 MB would be an immediate outlier. Deep learning checkpoints exhibit heavy-tailed size distributions: most tensors are small (biases, layer norms, scale factors), while a minority are large (embeddings, MLP up-projections, attention weight matrices). Constellation Encoding shards the payload across **many heterogeneous chunks**, where each chunk's size and dtype are inherited directly from whichever donor tensor was selected for it by the profile CSPRNG.

**Archetype cloning rules:**

- Every chunk adopts the **exact shape** (`dims[]`) of its assigned donor tensor.
- Every chunk adopts the **exact data type** (`FLOAT` or `FLOAT16`) of its assigned donor tensor.
- Chunk sizes are not equalized. The natural variance in donor tensor shapes is preserved and treated as a feature rather than a defect.

This organic irregularity is intentional. A 100 MB payload might be distributed across chunks of 13 MB, 24 MB, 41 MB, 22 MB, and so on, reflecting the archetype distribution of the donor model. Uniform chunk sizes would themselves be a detectable pattern. Similarly, the dtype mix of the injected constellation mirrors the dtype mix of the model's real weight population: in a model containing both FP16 and FP32 tensors, some stego chunks will be FP16 and others FP32, exactly as one would expect from a legitimate checkpoint.

This dtype mixing also has a pleasant arithmetic property. If the donor pool is composed of roughly equal fractions of FP16 and FP32 elements, the disk overhead of the injected tensors averages to approximately 3 bytes per payload byte — between the FP16 extreme of 2 bytes and the FP32 extreme of 4 bytes. Any real model with a natural dtype mix will land somewhere in this range without any deliberate tuning.

### 4.2 Empirical Distribution Mimicry (ECDF Encoding)

The payload bytes, even after encryption, are uniformly distributed over {0, …, 255}. The goal is to map each byte to a float value such that the resulting tensor's **marginal distribution** matches the donor's empirical histogram. The approach builds a 256-entry lookup table by selecting actual order statistics from the donor tensor, guaranteeing that all table entries are valid, representable values in the donor's native dtype.

#### 4.2.1 Mathematical Preliminaries

Let a donor tensor contain $n$ scalar values. Sort them into order statistics:

$$x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$$

The empirical cumulative distribution function (ECDF) of the donor is:

$$\hat{F}_n(t) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{\{x_{(i)} \le t\}}$$

For the purposes of building the encoding lookup table, we do not interpolate between order statistics. Instead, we select actual values that exist in the donor's sorted population, as described in the next section. This guarantees that every table entry is a value the donor tensor genuinely contains, and therefore a value that is natively representable in the donor's dtype without any rounding.

#### 4.2.2 Order-Statistic Lookup Encoding

Because the input alphabet is 256 byte values, Stnx precomputes a lookup table of exactly 256 entries, each drawn directly from the sorted donor population. For each $k \in \{0, 1, \dots, 255\}$, compute a target index:

$$j_k = \left\lfloor \frac{(k + 0.5) \cdot n}{256} \right\rfloor$$

The initial candidate table entry is then:

$$v_k^{*} = x_{(j_k)}$$

**Uniqueness guarantee.** All 256 table entries must be distinct, because the decoding step requires an unambiguous inverse mapping. After computing all 256 candidates, scan the table from $k = 1$ to $k = 255$. If any $v_k^{*} = v_{k-1}$, advance the index $j_k$ to the smallest $j > j_{k-1}$ such that $x_{(j)} \ne v_{k-1}$, and set $v_k = x_{(j)}$. The final entry $v_k$ is then used as the table value, and the search pointer for the next iteration begins at $j + 1$.

**Eligibility condition.** If the donor tensor contains fewer than 256 distinct scalar values, it is impossible to construct a 256-entry table with all distinct values. Such a tensor is ineligible as a donor and is excluded from the profile. For any real-world weight tensor with $n \ge 1024$ elements drawn from a continuous-like distribution, this condition is effectively never triggered, but it must be checked explicitly and the tensor must be rejected if it fails.

**Why this is safe for both FP32 and FP16.** Because every table entry $v_k$ is an actual value taken from the donor tensor's stored population, it is by construction a value that the donor's dtype can represent exactly. For FP32 donors, the entries are FP32 values. For FP16 donors, the entries are FP16 values (read from the tensor and potentially widened to FP32 for computation, but representing only FP16-exact numbers). When the encoded float is written into the stego tensor at the donor's dtype, no rounding occurs. This is the key property that makes the approach dtype-agnostic without any special-casing.

**Encoding.** For each payload byte $b$:

$$f_b = v_b$$

The resulting float $f_b$ is written at the donor's dtype into the stego tensor. The marginal distribution of the stego tensor is the 256-point discretization of the donor's ECDF, sampled at quantile midpoints — statistically indistinguishable from a coarsely sampled view of the real weight population.

#### 4.2.3 Extraction via Exact Lookup

To decode, the extractor reconstructs the identical 256-entry table from the donor tensor using the same order-statistic procedure and search pointer logic described above. Because the table entries are actual values from the donor, and because the stego tensor was written at the same dtype, each extracted float $y$ is an exact member of the table $\{v_0, \dots, v_{255}\}$.

Decoding is therefore an exact lookup:

$$b = k \quad \text{such that} \quad v_k = y$$

This is unambiguous because all table entries are distinct by construction. No nearest-neighbour search, no approximation, and no tolerance threshold is required. The extractor reads the donor tensor's `data_type` field directly from its `TensorProto` to determine whether to build the table in FP32 or FP16 precision, ensuring the same computation is reproduced exactly.

#### 4.2.4 Padding and Final Chunk Handling

The last chunk's donor shape will generally demand more element slots than there are remaining payload bytes. The surplus slots are filled with **synthetic draws from the same ECDF**:

$$f_{\text{pad}} = v_{b_{\text{synth}}}$$

where $b_{\text{synth}}$ is a byte drawn from a CSPRNG seeded by $K_{\text{pad}}$, the dedicated padding subkey described in Section 6. The padding bytes are mapped through the same lookup table as payload bytes, producing floats that are drawn from the same 256-point distribution. Synthetic padding is therefore statistically indistinguishable from payload-bearing slots to any marginal distribution test.

Never fill surplus slots with zeros, the distribution mean, or any repeated value. Such fills produce detectable spikes in the tensor's value histogram that would trivially fail the statistical gate.

### 4.3 Shape Archetype Cloning

During profiling, Stnx catalogs every eligible tensor and groups them by shape family:

| Archetype | Typical Shape | Element Count | Mimics |
|-----------|---------------|---------------|--------|
| `mlp_up` | `[H_in, 4*H_out]` | Large | Gated MLP up-projection |
| `mlp_down` | `[4*H_out, H_in]` | Large | MLP down-projection |
| `attn_v` | `[H, H]` | Medium | Value projection |
| `bias_mid` | `[H]` | Small | LayerNorm or bias |
| `embed_tok` | `[V, H]` | Very large | Token embedding |

The profile CSPRNG draws donors from the full eligible pool, which spans all archetype families and both FP16 and FP32 tensors. No attempt is made to draw proportionally or to equalize dtype representation. The resulting chunk sequence reflects whatever distribution the donor model's weight population happens to have.

### 4.4 Naming and Taxonomy

Stego tensor names must be plausible and ordered. The tool generates a deterministic sequence via a CSPRNG seeded by $K_{\text{name}}$, a subkey derived from the passphrase. Example taxonomy families:

| Family | Template | Visual Plausibility |
|--------|----------|---------------------|
| EMA shadow | `_ema.shadow.<hex>.{:04}` | Exponential moving average weights |
| Optimizer state | `_optim.exp_avg_sq.<hex>.{:04}` | Adam second moments |
| Adapter shards | `lora_B.<hex>.{:04}` | Low-rank adapter fragments |
| RoPE cache | `rotary_emb.inv_freq_ext_<hex>.{:04}` | Extended positional cache |

The `<hex>` component is an 8-character deterministic suffix derived from the CSPRNG; the index is zero-padded to four digits. Names are generated in CSPRNG order and checked against the set of all existing tensor names in the model. If a generated name collides with an existing tensor, the generator is advanced until a collision-free name is produced. This collision-advancement sequence is reproduced identically during extraction, so the extractor generates the same name sequence and maps any matching model tensor to a stego chunk without any out-of-band index.

---

## 5. Cryptographic Framing and Stream Layout

Before encoding, the payload undergoes a fixed preparation pipeline. The Argon2id salt is derived deterministically from the passphrase rather than generated randomly and stored; this avoids a bootstrap problem where the extractor would need the salt to derive the key, but would need the key to find the first stego chunk. The AES-256-GCM nonce, however, must be random and fresh at every injection, and is stored as the first bytes of the assembled payload stream.

```
Raw File
    │
    ▼
[zstd compress, level configurable (default 3)]
    │
    ▼
[172-byte plaintext header]
  - 4 B  format version (0x00000001)
  - 8 B  uncompressed file length (uint64 LE)
  - 32 B SHA-256 of raw file
  - 128 B original filename (null-terminated, remainder 0x00)
    │
    ▼ (header is prepended to the compressed payload byte stream)
    │
    ▼
[AES-256-GCM encrypt]
  - Key: K_enc derived from master secret (see Section 6)
  - Nonce: 12 B, cryptographically random, generated fresh at each injection
  - Tag: 16 B, appended after ciphertext
    │
    ▼
[assembled payload stream fed to Constellation Encoding]
  - 12 B  AES-256-GCM nonce (plaintext)
  - 8 B   ciphertext+tag byte count (uint64 LE)
  - N B   ciphertext
  - 16 B  GCM authentication tag
```

The 8-byte length field allows the extractor to know exactly how many bytes to accumulate before attempting decryption, without any out-of-band chunk count. The nonce field allows decryption to proceed immediately once all ciphertext bytes have been decoded from the stego tensors.

The 128-byte filename field accommodates typical filenames and moderately deep relative paths. The field is null-terminated; unused bytes are zeroed. Filenames longer than 127 bytes (leaving one byte for the null terminator) are truncated before storage. The SHA-256 field covers the original, pre-compression file bytes, providing integrity verification independent of the compression and encryption layers.

---

## 6. Deterministic Retrieval without External Metadata

Extraction must succeed with **only** the stego `.onnx` file and the passphrase. No manifest, no index, no JSON.

**Key derivation.** A deterministic salt is first computed from the passphrase using a cheap, domain-separated hash:

$$\text{salt} = \text{SHA-256}\!\left(\texttt{"stnx.kdf.v1"} \,\|\, \text{UTF-8}(\text{passphrase})\right)[{:16}]$$

The master secret $K$ is then derived via Argon2id with fixed, specified parameters:

$$K = \text{Argon2id}\!\left(\text{password} = \text{passphrase},\ \text{salt} = \text{salt},\ m = 65536\ \text{KiB},\ t = 3,\ p = 4,\ \text{taglen} = 32\right)$$

The parameters $m = 65536\ \text{KiB}$ (64 MiB), $t = 3$ iterations, and $p = 4$ lanes correspond to the RFC 9106 high-memory recommendation and provide strong resistance to GPU-based brute-force attacks. From $K$, four independent subkeys are derived via keyed hashing:

$$K_{\text{enc}} = \text{HMAC-SHA256}(K,\ \texttt{"stnx.enc"})$$
$$K_{\text{name}} = \text{HMAC-SHA256}(K,\ \texttt{"stnx.name"})$$
$$K_{\text{profile}} = \text{HMAC-SHA256}(K,\ \texttt{"stnx.profile"})$$
$$K_{\text{pad}} = \text{HMAC-SHA256}(K,\ \texttt{"stnx.pad"})$$

Their roles:

| Subkey | Purpose |
|--------|---------|
| $K_{\text{enc}}$ | AES-256-GCM encryption and decryption |
| $K_{\text{name}}$ | CSPRNG seed for generating stego tensor names |
| $K_{\text{profile}}$ | CSPRNG seed for selecting the donor sequence and chunk ordering |
| $K_{\text{pad}}$ | CSPRNG seed for generating synthetic padding bytes in the final chunk |

**Extraction algorithm:**

1. Load the stego model; enumerate all `initializer` names into a set $M$.
2. Derive the salt from the passphrase and compute $K$ via Argon2id. Derive all four subkeys.
3. Expand the name CSPRNG seeded by $K_{\text{name}}$ to produce candidate stego names, advancing past any collision with existing names as was done during injection. Any name in $M$ that matches a candidate is added to the stego set $S$; the remaining names constitute the donor pool $R$.
4. Expand the profile CSPRNG seeded by $K_{\text{profile}}$ to produce a deterministic ordering of donors $D_0, D_1, \dots$ drawn from $R$, skipping any donor with fewer than 1024 elements or fewer than 256 distinct values.
5. For each donor $D_i$, read its `data_type` field from the model to determine whether to build the ECDF table in FP32 or FP16 precision. Construct the 256-entry order-statistic table for $D_i$ as described in Section 4.2.2.
6. Decode chunk $S_0$ using $D_0$'s table to recover raw bytes.
7. The first 12 decoded bytes are the AES-256-GCM nonce. Bytes 13–20 (uint64 LE) are the stream length $L_{\text{stream}}$, counting ciphertext bytes plus the 16-byte GCM tag.
8. Continue decoding $S_1, S_2, \dots$ with donors $D_1, D_2, \dots$ until at least $L_{\text{stream}} + 20$ total bytes have been accumulated (20 = 12 nonce + 8 length).
9. Slice exactly $L_{\text{stream}}$ bytes following the nonce and length fields; any remainder is padding and is discarded.
10. Decrypt the slice via AES-256-GCM under $K_{\text{enc}}$ and the recovered nonce. Verify the GCM authentication tag; fail with an explicit tamper or wrong-passphrase error if verification fails.
11. Verify the SHA-256 of the decrypted, decompressed content against the value stored in the 172-byte header. Decompress with zstd. Write the recovered file to disk using the filename from the header.

Because the same passphrase deterministically regenerates the identical $S$ and $D$ sequences, chunk alignment is recovered automatically with no external state.

---

## 7. Operational Workflow

### 7.1 Profiling Phase

**Input:** path to donor `.onnx`.

The tool scans the model and builds an **Archetype Database**:

- Iterate `graph.initializer`.
- Retain only tensors with `data_type == FLOAT` (FP32) or `data_type == FLOAT16` (FP16). All other data types are skipped unconditionally.
- Skip tensors with fewer than 1024 elements (insufficient population for a 256-entry distinct-value table).
- For each retained tensor, verify that at least 256 distinct scalar values exist. Tensors failing this check are excluded.
- Record `(name, dims, data_type, sorted_values)` into archetype bins by shape family.
- Compute total eligible capacity: $C = \sum n_i$ across all eligible tensors, regardless of dtype.

Output: an in-memory capacity report printed to stderr, for example:

```
Profile complete.
Eligible FP32 tensors : 78    (total elements: 214,500,000)
Eligible FP16 tensors : 64    (total elements: 172,920,489)
Combined eligible elements   : 387,420,489
Effective disk overhead range: 2x – 4x payload bytes (depends on dtype mix)
Projected capacity @ 70%     : ~258.8 MB payload
Archetype distribution       : mlp_up(24), mlp_down(24), attn_qkv(72), bias(22)
```

If the projected capacity is insufficient for the requested payload, the tool exits before any mutation.

### 7.2 Injection Phase

**Input:**

- Donor model path
- Payload file path (arbitrary bytes: images, video, archives, executables)
- Passphrase (via stdin interactive prompt or secure environment variable)
- `--zstd-level` override (default 3)
- `--out` output path override

**Process:**

1. Derive the salt from the passphrase, compute $K$ and all four subkeys.
2. Generate a fresh 12-byte cryptographically random AES-256-GCM nonce.
3. Read and frame the payload as specified in Section 5.
4. Determine chunk count $m$ such that the total element demand across all $m$ chunks does not exceed $\alpha \cdot N_{\text{eligible}}$.
5. Generate stego names $S_0 \dots S_{m-1}$ via the $K_{\text{name}}$ CSPRNG, advancing past any collision with existing tensor names.
6. For each chunk $i$:
   - Select donor $D_i$ via the $K_{\text{profile}}$ CSPRNG from the eligible pool (both FP16 and FP32 donors participate).
   - Read $D_i$'s `data_type` and build the 256-entry order-statistic lookup table.
   - Map the next $|D_i|$ payload bytes to floats via the table. For payload bytes beyond the end of the ciphertext stream, draw synthetic bytes from the $K_{\text{pad}}$ CSPRNG and map them through the same table.
   - Serialize the result as a `TensorProto` with $D_i$'s `dims`, $D_i$'s `data_type`, and name $S_i$.
7. Append all $m$ new `TensorProto` entries to `graph.initializer`.
8. Run verification gates (Section 10).
9. Serialize to the output path.

**Output:** stego `.onnx` file.

### 7.3 Extraction Phase

**Input:**

- Stego model path
- Passphrase

**Process:** as described in Section 6.

**Output:** recovered original file written to disk with the filename embedded in the decrypted header.

---

## 8. Interface Specification

Stnx is a CLI tool exposing three conceptual stages. The exact command names are illustrative.

| Stage | Primary Inputs | Primary Outputs | Failure Modes |
|-------|----------------|-----------------|---------------|
| `profile` | Donor `.onnx` path | Capacity report (human readable); exit code 0 or 1 | Invalid protobuf; no eligible FP16/FP32 tensors; all tensors too small or insufficiently distinct |
| `inject` | Donor `.onnx`, payload file, passphrase, optional `--zstd-level`, optional `--out` | Stego `.onnx` | Capacity exceeded; no eligible archetypes; name collision exhaustion; statistical self-test failure |
| `extract` | Stego `.onnx`, passphrase | Recovered file to filesystem | Wrong passphrase (K_name → no matching names found); GCM tag mismatch (tampered model); SHA-256 mismatch (corrupted payload); zstd decompress error |
| `verify` | Stego `.onnx`, passphrase | Per-chunk K–S report; entropy report | Same failure modes as `extract`; statistical alarm if any chunk diverges from its donor ECDF |

The `verify` command performs structural validation and per-chunk statistical analysis only. It does not load either model into ONNX Runtime and performs no inference. This keeps the tool dependency-free with respect to the runtime and is consistent with the intended use case of stored rather than executed models.

All file I/O is streaming where possible; models larger than RAM are supported via memory-mapped deserialization if the protobuf implementation permits.

---

## 9. Capacity Analysis and Sizing Formulas

### 9.1 Fundamental Constraint

Let the donor model contain $N$ eligible scalar elements across all usable FP16 and FP32 initializers. Because Constellation Encoding stores exactly **one payload byte per float element**, the raw byte capacity is $N$ bytes, regardless of the element's dtype.

Stnx conservatively commits only a fraction $\alpha = 0.70$ of available elements to avoid packing the model so densely with unused tensors that their count alone becomes anomalous. The **guaranteed payload capacity** is:

$$C_{\text{max}} = \lfloor \alpha N \rfloor \quad \text{bytes}$$

Given a target payload of $L$ bytes (after compression and encryption overhead), the minimum required donor model in total eligible elements is:

$$N_{\text{min}} = \left\lceil \frac{L}{\alpha} \right\rceil$$

### 9.2 Disk Size Relationship

ONNX stores FP16 scalars as 2 bytes each and FP32 scalars as 4 bytes each. The disk overhead of the injected constellation depends on the dtype mix of the donors selected by the profile CSPRNG. Let $N_{16}$ and $N_{32}$ be the number of FP16 and FP32 elements consumed by the payload respectively, with $N_{16} + N_{32} = L$ (ignoring the padding tax). The effective disk overhead multiplier is:

$$\bar{d} = \frac{2 N_{16} + 4 N_{32}}{N_{16} + N_{32}}$$

This value ranges from 2.0 (all FP16 donors) to 4.0 (all FP32 donors) and averages to approximately 3.0 for a balanced mixed-precision model. The total disk size increase of the stego file relative to the donor is:

$$\Delta_{\text{disk}} \approx \bar{d} \cdot L$$

In practice, $\bar{d}$ is determined by whatever the donor model's dtype distribution happens to be and is reported by the profiler before injection.

### 9.3 Capacity Reference Table

Assuming $\alpha = 0.70$ and that the entire eligible weight budget is utilized:

| Total Donor Params ($N$) | Eligible Elements ($0.7N$) | Max Payload $L$ | FP16-only Δ disk | FP32-only Δ disk |
|--------------------------|----------------------------|-----------------|------------------|------------------|
| 10 M  | 7.0 M   | **6.7 MB**    | ~13 MB   | ~27 MB   |
| 50 M  | 35.0 M  | **33.4 MB**   | ~67 MB   | ~134 MB  |
| 100 M | 70.0 M  | **66.8 MB**   | ~134 MB  | ~268 MB  |
| 200 M | 140.0 M | **133.6 MB**  | ~267 MB  | ~534 MB  |
| 400 M | 280.0 M | **267.0 MB**  | ~534 MB  | ~1.07 GB |
| 1 B   | 700.0 M | **667.6 MB**  | ~1.34 GB | ~2.67 GB |

The Δ disk columns show the extremes. A real mixed-precision model lands between them.

### 9.4 Inverse Sizing: Required Model for Given Payload

Given a desired payload size $L$ (MB), the minimum donor model parameter count and approximate disk footprint are:

| Payload $L$ | Min Params ($N_{\text{min}}$) | FP16-only Δ disk | FP32-only Δ disk |
|-------------|-------------------------------|------------------|------------------|
| 5 MB   | 7.5 M   | ~15 MB   | ~30 MB   |
| 25 MB  | 37.8 M  | ~76 MB   | ~151 MB  |
| 50 MB  | 75.5 M  | ~151 MB  | ~302 MB  |
| 100 MB | 149.8 M | ~300 MB  | ~600 MB  |
| 200 MB | 298.6 M | ~597 MB  | ~1.19 GB |
| 500 MB | 746.7 M | ~1.49 GB | ~2.99 GB |

*These figures assume the model is entirely composed of eligible weights. Real models contain metadata, smaller bias tensors below the 1024-element floor, ineligible quantized layers, and graph description overhead. Add a 10–15% headroom margin in practice.*

---

## 10. Verification Gates

Before emitting the final `.onnx`, Stnx must pass two gates. Failures are fatal and no output file is written.

### 10.1 Structural Gate

- Run the equivalent of `onnx.checker.check_model(model)`.
- Assert the graph remains acyclic, all node references resolve to valid types, and all stego tensors are well-formed `TensorProto` entries with valid `data_type`, non-empty `dims`, and `raw_data` of the correct byte length.
- Assert that no stego tensor name collides with any node input or output name (verifying that the stego tensors remain unreferenced by the graph).

### 10.2 Statistical Gate

For every injected chunk $i$ with encoded floats $y_1, \dots, y_m$ and its donor $D_i$:

- **Two-sample Kolmogorov–Smirnov test** between the stego tensor's float values and the donor's sorted population. The null hypothesis (same distribution) must not be rejected at $\alpha = 0.05$.
- **Chi-squared byte-frequency test** on the serialized `raw_data` of the injected tensor versus a simulated sample of the same shape drawn from the same order-statistic table. Because the bytes are little-endian packed floats, the marginal byte frequencies of the stego tensor's raw bytes should be consistent with what a genuine weight tensor of that shape would produce.

If either sub-test rejects any chunk, the injection is aborted and an error is reported identifying the failing chunk and its donor.

---

## 11. Internal Limitations

Stnx operates under constraints that are architectural rather than implementation deficiencies. Understanding them is important for sizing payloads and selecting appropriate donor models.

### 11.1 Quantized and Discrete-Format Rejection

Only `FLOAT` (FP32) and `FLOAT16` (FP16) tensors are eligible as donors. Block-quantized formats — `INT4`, `QINT8` with per-block scales, `UINT8` lookup tables — are rejected unconditionally during profiling. The reason is that their discrete, low-cardinality value spaces make ECDF-based distribution mimicry impossible: a 4-bit weight can assume only 16 distinct values, and encoding 256 payload bytes into a domain of 16 symbols requires massive value duplication that is immediately detectable as a histogram anomaly.

Any model whose weight population is composed entirely of such formats cannot serve as a donor and will produce a `CapacityError` at the profiling phase.

### 11.2 ECDF Distinctness Floor

A donor tensor must satisfy two conditions: at least 1024 elements (to provide a sufficiently dense sorted population from which 256 spaced order statistics can be drawn), and at least 256 distinct scalar values (to guarantee the uniqueness requirement of the lookup table). For any real-valued weight tensor in a trained neural network with $n \ge 1024$ elements, the 256-distinct-values condition is virtually always satisfied; the 1024-element floor is the operative gate in practice.

Tensors failing either condition are demoted to non-donor status and do not contribute to the capacity pool, but they do remain in the model's `initializer[]` array untouched. They continue to participate as the donor pool $R$ from which the profile CSPRNG draws during extraction, so their presence must be stable between injection and extraction.

### 11.3 Capacity Is Bounded by the Donor

The tool cannot manufacture capacity. If the payload — after zstd compression and AES-256-GCM framing — exceeds $\alpha \cdot N_{\text{eligible}}$ bytes, injection is impossible and the tool exits with a `CapacityError` before modifying any file. Users on the boundary should try a higher zstd compression level or obtain a larger donor model.

### 11.4 Deterministic Dependence on Donor Content

Extraction depends on being able to reconstruct the donor sequence $D_0, D_1, \dots$ from the pool of non-stego tensors remaining in the stego model. If a hypothetical adversary strips or renames real weight tensors while leaving the stego tensors intact, the extracted $D_i$ sequence becomes desynchronized from the sequence used during injection, and decoding produces garbage. The AES-256-GCM tag verification will catch this and report a tamper error. This is expected behaviour; structural modification of the model invalidates extraction.

### 11.5 File Hash Mutation

Any modification to `graph.initializer` necessarily changes the serialized protobuf bytes, producing a different SHA-256 hash for the `.onnx` file. Stnx does not attempt to preserve the original file hash — doing so would require altering existing tensors, which violates the functional invariance guarantee. Redistribution of a stego model should account for this by presenting the file as a fine-tuned variant, an EMA checkpoint, or a re-exported model (for example, `model-fp16-ema.onnx`).

### 11.6 Padding Tax

The final chunk is padded to its full donor shape. The padding elements consume capacity from the $\alpha N$ budget but carry no payload information. For a final chunk of size $M$ with $P$ remaining payload bytes, the wasted capacity is $M - P$ elements. For large chunk sizes (e.g., an embedding table donor with millions of elements as the last chunk), this tax can be substantial. Stnx does not attempt to minimize the padding tax by reordering chunks; the profile CSPRNG ordering is fixed by $K_{\text{profile}}$ and must be reproduced identically during extraction.

### 11.7 No Selective Tensor Update

Stnx only appends initializers. It does not modify existing weights, and there is no `uninject` operation that restores the original file hash. To clean a model, one must re-export it from the original source or strip the stego tensors by removing all initializer entries whose names match the $K_{\text{name}}$ CSPRNG sequence, which requires the passphrase.

### 11.8 Memory Pressure During Profiling

Computing order statistics for the ECDF requires loading eligible donor tensors into memory as sortable arrays of scalar values. For a large mixed-precision model with hundreds of millions of parameters, this may temporarily require several gigabytes of working memory. FP16 tensors are widened to FP32 for the sort step, doubling their in-memory footprint relative to on-disk size. Streaming sort algorithms are not used in the reference design; users on memory-constrained systems should expect proportional allocation during the profiling phase.
