# DFT / Type-II-Inspired Codebook-Based MU-MIMO Precoding over OFDM

Limited-feedback zero-forcing precoding for a MU-MIMO downlink.

Instead of reporting perfect channel state information (CSI), each user feeds back a quantized channel direction through a Precoder Matrix Indicator (PMI). The base station then constructs a zero-forcing precoder from those quantized directions.

This project studies how limited feedback affects:

- channel-direction representation
- MU-MIMO sum rate
- QPSK BER
- PMI collisions
- single-beam vs multi-beam feedback

The multi-beam experiment is **Type-II-inspired**, not a complete implementation of the 3GPP Type II codebook. It uses an oversampled DFT beam grid with quantized amplitude and relative-phase coefficients to study the advantage of representing a multipath channel with more than one beam.

---

## Features

- Oversampled DFT codebook for PMI feedback
- Geometric clustered-multipath channel model
- Perfect-CSI and quantized-CSI zero-forcing precoding
- QPSK Monte Carlo BER simulation
- Sum-rate analysis
- PMI collision analysis
- Single-beam vs Type-II-inspired multi-beam comparison
- Reproducible Monte Carlo experiments

---

## Repository Structure

```text
mu-mimo-codebook-precoding/
├── figures/
│   ├── experiment_1.png
│   ├── experiment_2.png
│   ├── experiment_3.png
│   └── experiment_4.png
├── BLOG.md
├── LICENSE
├── README.md
├── experiment_4.py
├── experiments_1_2_3.py
├── precoding.py
├── requirements.txt
├── run_all.py
└── .gitignore
```

### Main Files

- `precoding.py` — channel model, DFT codebook, PMI selection, Type-II-inspired multi-beam approximation, ZF precoder, sum-rate calculation, and QPSK BER simulation.
- `experiments_1_2_3.py` — match quality, sum-rate, and BER / PMI-collision experiments.
- `experiment_4.py` — single-beam vs multi-beam feedback comparison.
- `run_all.py` — runs all experiments and generates the figures.
- `BLOG.md` — detailed technical write-up covering implementation reasoning, debugging, and interpretation of the results.

---

## System Model

For user `k`, the received signal is

```text
y_k = h_k^H x + n_k
```

with

```text
x = Σ_i w_i s_i
```

where:

- `h_k` — channel vector between the BS antenna array and user `k`
- `w_i` — precoding vector for user `i`
- `s_i` — transmitted QPSK symbol
- `n_k` — additive noise

The effective MU-MIMO channel is

```python
G = H.conj() @ W
```

---

## Channel Model

`geometric_channel()` generates a clustered multipath channel containing `n_paths` propagation paths distributed around one randomly selected angular center.

The main experiments use

```text
Nt = 4 antennas
n_paths = 3
```

representing a **3-path clustered multipath channel**.

---

## PMI Selection

Each user selects the DFT codeword that provides the strongest normalized channel-direction match.

The PMI metric is based on the Hermitian inner product

```text
|h^H c|²
```

where `h` is the channel vector and `c` is a candidate codeword.

---

## Zero-Forcing Precoding

Because the receiver model evaluates the effective gain as `h_k^H w`, the ZF precoder is constructed using the conjugated channel matrix:

```python
W = np.linalg.pinv(H.conj())
```

before per-column power normalization.

For perfect CSI,

```python
H.conj() @ W
```

is diagonal up to power normalization when the channel matrix has sufficient rank, suppressing inter-user interference.

---

## Type-II-Inspired Multi-Beam Representation

A single DFT codeword represents one dominant spatial direction.

The multi-beam model instead approximates the channel as a weighted combination of several selected DFT beams.

The strongest beam is used as the reference. Remaining coefficients are represented using:

- quantized amplitude
- quantized relative phase

This allows the feedback representation to capture multipath spatial structure that cannot be represented by a single beam alone.

---

# Experiments

## Experiment 1 — Match Quality vs Feedback Bits

The same 400 channel realizations are evaluated at every feedback-bit value.

| Bits | Single-path | 3-path clustered multipath |
|---:|---:|---:|
| 1 | 0.3846 | 0.3891 |
| 2 | 0.7532 | 0.7219 |
| 3 | 0.9404 | 0.8997 |
| 5 | 0.9961 | 0.9513 |
| 10 | 1.0000 | 0.9551 |

The single-path channel converges almost perfectly to one DFT beam as codebook resolution increases.

The multipath channel instead approaches a match-quality ceiling of approximately **0.955**, demonstrating the structural limitation of representing a multipath spatial channel with one beam.

![DFT codebook match quality](figures/experiment_1.png)

---

## Experiment 2 — Sum Rate vs Feedback Bits

Configuration:

```text
K = Nt = 4
SNR = 20 dB
300 channel realizations
```

| Bits/user | Mean sum rate (bits/s/Hz) |
|---:|---:|
| 1 | 2.830 |
| 2 | 5.633 |
| 3 | 8.176 |
| 4 | 9.199 |
| 5 | 9.535 |
| 10 | 9.390 |
| Perfect CSI | 11.665 |

Performance improves rapidly as feedback increases from 1 to approximately 5 bits.

Beyond that point, the single-beam representation remains around **9.4–9.5 bits/s/Hz**, while perfect CSI reaches **11.665 bits/s/Hz**.

Small non-monotonic variations can occur because fully loaded ZF depends on the **joint conditioning of all quantized user directions**, not only the individual PMI match quality.

![Sum rate versus feedback bits](figures/experiment_2.png)

---

## Experiment 3 — BER and PMI Collision

A fixed, well-conditioned MU-MIMO channel is used to isolate the effect of PMI quantization.

The experiment compares:

- Perfect CSI
- 1-bit PMI
- 3-bit PMI
- 6-bit PMI

Perfect-CSI ZF reaches zero observed BER at high SNR.

The 3-bit and 6-bit cases retain waterfall-shaped BER behavior and eventually reach zero observed errors in the Monte Carlo simulation.

With only 1-bit feedback, multiple users select the same PMI. This **PMI collision** makes the quantized MU-MIMO channel rank deficient and prevents complete user separation.

The resulting aggregate BER approaches approximately **0.25** at high SNR.

This is not equivalent to fully random bit guessing, which would correspond to BER = 0.5.

![BER versus SNR with PMI collision](figures/experiment_3.png)

---

## Experiment 4 — Single-Beam vs Type-II-Inspired Multi-Beam Feedback

### Channel Match Quality

| Configuration | Bits/user | Match quality |
|---|---:|---:|
| Type I single beam | 13 | 0.9572 |
| Type-II-inspired, L=2 | 13 | 0.9774 |
| Type-II-inspired, L=3 | 22 | 0.9812 |
| Type-II-inspired, L=3 richer coefficients | 26 | 0.9945 |

At the same **13-bit feedback budget**, the two-beam representation improves channel match quality from

```text
0.9572 → 0.9774
```

### Sum Rate at the Matched 13-Bit Budget

| Scheme | Mean sum rate (bits/s/Hz) |
|---|---:|
| Type I, 13 bits | 9.072 |
| Type-II-inspired L=2, 13 bits | 9.467 |
| Perfect CSI | 11.529 |

The multi-beam representation therefore recovers part of the performance lost by the single-beam codebook without increasing the feedback budget.

For additional context, the richer 26-bit multi-beam configuration reaches

```text
Match quality = 0.9945
Sum rate      = 10.599 bits/s/Hz
```

This uses a larger feedback budget and is therefore not a direct equal-bit comparison.

![Single-beam versus multi-beam feedback](figures/experiment_4.png)

---

## Main Observations

1. **Codebook resolution**  
   Increasing PMI bits significantly improves angular resolution at low feedback budgets.

2. **Single-beam representation limit**  
   Increasing single-beam PMI resolution cannot fully represent a clustered multipath spatial channel.

3. **Multi-beam feedback**  
   Combining multiple beams with amplitude and relative-phase information improves both channel representation and MU-MIMO sum rate.

4. **Different metrics reveal different effects**
   - Match quality measures channel representation error.
   - Sum rate captures MU-MIMO performance loss continuously.
   - BER exposes severe PMI-collision behavior at very coarse feedback resolution.

---

## Running the Experiments

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run all experiments:

```bash
python3 run_all.py
```

Generated figures are written to:

```text
figures/
```

---

## Requirements

- Python 3
- NumPy
- Matplotlib

---

## Technical Write-Up

For a deeper discussion of the implementation, debugging process, and interpretation of the experiments, see [`BLOG.md`](BLOG.md).

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
