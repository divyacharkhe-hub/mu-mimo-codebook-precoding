# DFT / Type-II-Inspired Codebook-Based MU-MIMO Precoding over OFDM

Limited-feedback zero-forcing precoding for a MU-MIMO downlink: users report a
quantized channel direction (PMI) instead of full CSI, the base station builds
a ZF precoder from those quantized directions, and the experiments measure how
much performance is lost to feedback quantization versus perfect CSI.

The multi-beam experiment is **Type-II-inspired**, not a full implementation of
3GPP Type II. It uses an oversampled DFT beam grid plus quantized amplitude and
relative-phase coefficients to test the structural advantage of representing a
channel with more than one beam.

## Files

- `precoding.py` — geometric clustered-multipath channel model, single-beam DFT
  codebook, Type-II-inspired multi-beam approximation, ZF precoder, sum-rate,
  and QPSK Monte Carlo BER simulator.
- `experiments_1_2_3.py` — match quality vs feedback bits, sum-rate ceiling,
  and BER vs SNR / PMI-collision behavior.
- `experiment_4.py` — single-beam vs multi-beam comparison, including a truly
  matched 13-bit feedback-budget sum-rate comparison.
- `run_all.py` — runs every experiment and moves plots into `figures/`.
- `requirements.txt` — Python dependencies.
- `BLOG.md` — write-up of the experiments, debugging, and verified conclusions.

## Channel model

`geometric_channel()` generates a **clustered multipath channel with `n_paths`
paths around one randomly selected angular center**. In the experiments below,
`n_paths=3`, so the correct description is **3-path clustered multipath**, not
"3 independent clusters."

## Key implementation choices

### 1. PMI inner-product convention

PMI selection must score a codeword with the Hermitian inner product
`|h^H c|^2`. The implementation now consistently evaluates that quantity.

### 2. ZF conjugate convention

The downlink model used throughout the code is

```text
y_k = h_k^H x + n_k
```

so the effective MU-MIMO channel is

```python
G = H.conj() @ W
```

The ZF precoder is therefore built from

```python
W = np.linalg.pinv(H.conj())
```

before column power normalization. This makes `H.conj() @ W` diagonal (up to
normalization) for perfect CSI, consistent with both `sum_rate()` and
`ber_sim()`.

### 3. Multi-beam relative phase

The first multi-beam implementation mixed a zero-phase reference beam with
**absolute** phases on the remaining beams. That is internally inconsistent.
The implementation uses the strongest selected beam as phase zero
and quantizes every other coefficient **relative to that reference phase**.

### 4. Fair Monte Carlo comparisons

Experiments 1 and 2 now reuse the **same channel realizations** at every
feedback-bit value. That removes avoidable Monte Carlo noise from comparisons
between nested codebooks and makes the match-quality trend monotonic as it
should be.

## Reproducible results

### Experiment 1 — match quality vs feedback bits

Using the same 400 channel draws at every bit count:

| bits | single-path | 3-path clustered multipath |
|---:|---:|---:|
| 1 | 0.3846 | 0.3891 |
| 2 | 0.7532 | 0.7219 |
| 3 | 0.9404 | 0.8997 |
| 5 | 0.9961 | 0.9513 |
| 10 | 1.0000 | 0.9551 |

The single-path channel converges to a single DFT beam. The 3-path clustered
channel reaches a structural single-beam plateau around 0.955.

### Experiment 2 — sum-rate ceiling, K=Nt=4, 20 dB

Using the same 300 channel matrices for every bit count and for perfect CSI:

| bits/user | mean sum-rate (bits/s/Hz) |
|---:|---:|
| 1 | 2.830 |
| 2 | 5.633 |
| 3 | 8.176 |
| 4 | 9.199 |
| 5 | 9.535 |
| 10 | 9.390 |
| perfect CSI | 11.665 |

The rate rises quickly through roughly 5 bits and then stays in the ~9.4–9.5
bits/s/Hz region. Small non-monotonic movement in sum-rate is possible even
with nested per-user codebooks because fully-loaded ZF depends on the **joint
conditioning** of all quantized user directions, not only each user's separate
match score.

### Experiment 3 — BER threshold / PMI collision

For the fixed well-conditioned channel draw:

- Perfect-CSI ZF reaches zero observed BER at high SNR.
- 3-bit and 6-bit PMI follow waterfall-shaped BER curves and eventually reach
  zero observed errors in the Monte Carlo run.
- The 1-bit case contains a **PMI collision**. Its aggregate BER approaches
  ~0.25 at high SNR because the quantized MU-MIMO channel becomes rank
  deficient and only part of the user separation survives.

`BER≈0.25` is **not** "random guessing" in the general sense; fully random bit
guessing would be BER=0.5.

### Experiment 4 — single-beam vs Type-II-inspired multi-beam

Match quality:

| configuration | bits/user | match quality |
|---|---:|---:|
| Type I single beam | 13 | 0.9572 |
| Type-II-inspired, L=2 | 13 | 0.9774 |
| Type-II-inspired, L=3 | 22 | 0.9812 |
| Type-II-inspired, L=3 richer coefficients | 26 | 0.9945 |

At the **actually matched 13-bit budget**:

| scheme | mean sum-rate (bits/s/Hz) |
|---|---:|
| Type I, 13 bits | 9.072 |
| Type-II-inspired L=2, 13 bits | 9.467 |
| perfect CSI | 11.529 |

For context only — **not an equal-feedback comparison** — the richer 26-bit
multi-beam configuration reaches 10.599 bits/s/Hz and 0.9945 match quality.

## Running

```bash
pip install -r requirements.txt
python3 run_all.py
```

The plots are written to `figures/`.
