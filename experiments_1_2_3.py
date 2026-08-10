import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from precoding import (
    geometric_channel, dft_codebook, pmi_select,
    zf_precoder, quantized_precoder, sum_rate, ber_sim,
)

Nt = 4
rng = np.random.default_rng(1)


def match_quality(h, codebook):
    _, cw = pmi_select(h, codebook)
    num = np.abs(cw @ h.conj()) ** 2
    den = (np.linalg.norm(h) ** 2) * (np.linalg.norm(cw) ** 2)
    return num / den


# ======================================================================
# Experiment 1: codeword match quality vs feedback bits.
# IMPORTANT: every feedback-bit value is evaluated on the SAME channel
# draws. Because these oversampled DFT codebooks are nested, the match
# quality should then be monotonic (up to numerical precision).
# ======================================================================
bit_range = list(range(1, 11))
n_trials = 400
mq_results = {1: [], 3: []}

channel_sets = {
    n_paths: [geometric_channel(Nt, n_paths, rng) for _ in range(n_trials)]
    for n_paths in (1, 3)
}

for n_paths in (1, 3):
    channels = channel_sets[n_paths]
    for bits in bit_range:
        cb = dft_codebook(Nt, bits)
        q = [match_quality(h, cb) for h in channels]
        mq_results[n_paths].append(np.mean(q))

print("=== Experiment 1: match quality vs bits ===")
print("bits        :", bit_range)
print("single-path:", np.round(mq_results[1], 4))
print("3-path     :", np.round(mq_results[3], 4))

plt.figure(figsize=(6, 4.5))
plt.plot(bit_range, mq_results[1], "o-", label="single-path (LOS)")
plt.plot(bit_range, mq_results[3], "s-", label="clustered multipath (3 paths)")
plt.xlabel("Feedback bits")
plt.ylabel("Normalized match quality  |h^H c|² / (‖h‖²‖c‖²)")
plt.title("DFT codebook match quality: single beam vs clustered multipath")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./experiment_1.png", dpi=150)
plt.close()


# ======================================================================
# Experiment 2: sum-rate ceiling vs feedback bits, fully-loaded K=Nt.
# Use the SAME channel realizations for every feedback-bit value and for
# the perfect-CSI reference so differences are due to the codebook, not
# independent Monte Carlo draws.
# ======================================================================
K = Nt
n_paths_e2 = 3
noise_power = 10 ** (-20 / 10)   # 20 dB operating SNR
n_avg = 300

sr_bits = list(range(1, 11))
H_draws = [
    np.stack([geometric_channel(Nt, n_paths_e2, rng) for _ in range(K)])
    for _ in range(n_avg)
]

sr_means = []
for bits in sr_bits:
    cb = dft_codebook(Nt, bits)
    rates = []
    for H_true in H_draws:
        W_q, _ = quantized_precoder(H_true, cb)
        rates.append(sum_rate(H_true, W_q, noise_power))
    sr_means.append(np.mean(rates))

perfect_rates = []
for H_true in H_draws:
    W_p = zf_precoder(H_true)
    perfect_rates.append(sum_rate(H_true, W_p, noise_power))
perfect_mean = np.mean(perfect_rates)

print("\n=== Experiment 2: sum-rate vs feedback bits (K=Nt=4, 20dB) ===")
print("bits vs mean sum-rate:", list(zip(sr_bits, np.round(sr_means, 3))))
print("perfect-CSI mean sum-rate:", round(perfect_mean, 3))

plt.figure(figsize=(6.5, 4.5))
plt.plot(sr_bits, sr_means, "o-", label="quantized PMI (ZF)")
plt.axhline(perfect_mean, color="gray", ls="--", label="perfect CSI (ZF)")
plt.xlabel("Feedback bits per user")
plt.ylabel("Sum rate (bits/s/Hz)")
plt.title(f"Sum-rate ceiling from PMI quantization, K=Nt={Nt}, 20 dB")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./experiment_2.png", dpi=150)
plt.close()


# ======================================================================
# Experiment 3: BER vs SNR -- threshold/collapse behavior.
# At very coarse feedback, duplicate PMIs make the quantized channel
# rank-deficient. In this particular fixed draw the aggregate BER sits
# near 0.25 at high SNR; that is NOT universal "random guessing" (which
# would be BER=0.5), but an average over users when only part of the MU-
# MIMO separation has collapsed.
# ======================================================================
snr_range = list(range(0, 31, 3))
n_symbols = 200000
rng2 = np.random.default_rng(9)   # well-conditioned draw (cond(H)~1.8)
H_fixed = np.stack([geometric_channel(Nt, n_paths_e2, rng2) for _ in range(K)])

W_perfect = zf_precoder(H_fixed)
curves = {"perfect CSI": [ber_sim(H_fixed, W_perfect, s, n_symbols, rng2) for s in snr_range]}

for bits in [1, 3, 6]:
    cb = dft_codebook(Nt, bits)
    W_q, pmis = quantized_precoder(H_fixed, cb)
    collapsed = len(set(pmis)) < K
    label = f"{bits}-bit PMI" + (" (PMI collision)" if collapsed else "")
    curves[label] = [ber_sim(H_fixed, W_q, s, n_symbols, rng2) for s in snr_range]

print("\n=== Experiment 3: BER vs SNR ===")
for label, ber in curves.items():
    print(f"{label:24s}:", np.round(ber, 5))

plt.figure(figsize=(6.5, 4.5))
for label, ber in curves.items():
    plt.semilogy(snr_range, np.maximum(ber, 1e-6), "o-", label=label)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"QPSK BER: PMI collision creates a threshold effect (K=Nt={Nt})")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("./experiment_3.png", dpi=150)
plt.close()

print("\nSaved experiment_1.png, experiment_2.png, experiment_3.png")
