import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from precoding import (
    geometric_channel, dft_codebook, pmi_select,
    oversampled_beams, type2_direction, type2_precoder_inputs,
    zf_precoder, sum_rate,
)

Nt = 4
n_paths = 3
rng = np.random.default_rng(3)


def match_quality(h, h_hat):
    num = np.abs(h_hat @ h.conj()) ** 2
    den = (np.linalg.norm(h) ** 2) * (np.linalg.norm(h_hat) ** 2)
    return num / den


# ----------------------------------------------------------------
# Type I: single-beam DFT codebook. Use the same channel draws for
# every bit count so the nested-codebook trend is directly comparable.
# ----------------------------------------------------------------
n_trials = 400
channels = [geometric_channel(Nt, n_paths, rng) for _ in range(n_trials)]
type1_bits = list(range(2, 14))  # include 13 for an exact L=2 budget match
type1_mq = []
for bits in type1_bits:
    cb = dft_codebook(Nt, bits)
    q = []
    for h in channels:
        _, cw = pmi_select(h, cb)
        q.append(match_quality(h, cw))
    type1_mq.append(np.mean(q))

# ----------------------------------------------------------------
# Type-II-inspired multi-beam representation. Coefficient phases are
# quantized RELATIVE to the strongest selected beam (the reference).
# This is intentionally simplified, not a full 3GPP Type II codebook.
# ----------------------------------------------------------------
beams = oversampled_beams(Nt, oversample=4)   # 16 beams -> 4 index bits each

type2_configs = [
    dict(L=2, amp_bits=2, phase_bits=3),   # 13 bits/user
    dict(L=3, amp_bits=2, phase_bits=3),
    dict(L=3, amp_bits=3, phase_bits=4),
]
type2_results = []
for cfg in type2_configs:
    q, bits_list = [], []
    for h in channels:
        h_hat, bits = type2_direction(h, beams, **cfg)
        q.append(match_quality(h, h_hat))
        bits_list.append(bits)
    type2_results.append((np.mean(bits_list), np.mean(q), cfg))

print("=== Experiment 4: Type I vs Type-II-inspired match quality (3-path clustered multipath) ===")
print("Type I bits -> match quality:")
for b, m in zip(type1_bits, type1_mq):
    print(f"  {b:2d} bits -> {m:.4f}")
print("Type-II-inspired bits -> match quality:")
for bits, mq, cfg in type2_results:
    print(f"  ~{bits:.1f} bits -> {mq:.4f}   (L={cfg['L']}, amp_bits={cfg['amp_bits']}, phase_bits={cfg['phase_bits']})")

plt.figure(figsize=(6.5, 4.8))
plt.plot(type1_bits, type1_mq, "o-", label="Type I (single beam)")
t2_bits = [b for b, _, _ in type2_results]
t2_mq = [m for _, m, _ in type2_results]
t2_labels = [f"L={c['L']}" for _, _, c in type2_results]
plt.plot(t2_bits, t2_mq, "^-", color="green", label="Type-II-inspired (multi-beam)")
for b, m, lab in zip(t2_bits, t2_mq, t2_labels):
    plt.annotate(lab, (b, m), textcoords="offset points", xytext=(6, -10))
plt.xlabel("Feedback bits per user")
plt.ylabel("Normalized match quality")
plt.title("Single-beam vs multi-beam feedback: 3-path clustered multipath")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./experiment_4.png", dpi=150)
plt.close()


# ----------------------------------------------------------------
# Sum-rate at an ACTUALLY MATCHED 13-bit budget: Type I 13 bits vs
# Type-II-inspired L=2 (13 bits), plus perfect CSI.
# ----------------------------------------------------------------
K = Nt
noise_power = 10 ** (-20 / 10)
n_avg = 250
matched_cfg = type2_configs[0]   # L=2, amp=2, phase=3 -> 13 bits/user
matched_bits = int(type2_results[0][0])
assert matched_bits == 13

H_draws = [
    np.stack([geometric_channel(Nt, n_paths, rng) for _ in range(K)])
    for _ in range(n_avg)
]

t1_rates, t2_rates, rich_t2_rates, perfect_rates = [], [], [], []
cb = dft_codebook(Nt, matched_bits)
rich_cfg = type2_configs[-1]
for H_true in H_draws:
    H_hat1 = np.zeros_like(H_true)
    for k in range(K):
        _, cw = pmi_select(H_true[k], cb)
        H_hat1[k] = cw
    W1 = zf_precoder(H_hat1)
    t1_rates.append(sum_rate(H_true, W1, noise_power))

    H_hat2, bits_used = type2_precoder_inputs(H_true, beams, **matched_cfg)
    assert bits_used == matched_bits
    W2 = zf_precoder(H_hat2)
    t2_rates.append(sum_rate(H_true, W2, noise_power))

    H_hat_rich, rich_bits_used = type2_precoder_inputs(H_true, beams, **rich_cfg)
    W_rich = zf_precoder(H_hat_rich)
    rich_t2_rates.append(sum_rate(H_true, W_rich, noise_power))

    Wp = zf_precoder(H_true)
    perfect_rates.append(sum_rate(H_true, Wp, noise_power))

print(f"\n=== Sum-rate at matched {matched_bits}-bit budget (K=Nt=4, 20dB) ===")
print(f"Type I            ({matched_bits} bits): {np.mean(t1_rates):.3f} bits/s/Hz")
print(f"Type-II-inspired  ({matched_bits} bits): {np.mean(t2_rates):.3f} bits/s/Hz")
print(f"perfect CSI                 : {np.mean(perfect_rates):.3f} bits/s/Hz")

# Also report the richer configuration as an UNMATCHED representation result.
rich_bits, rich_mq, rich_cfg = type2_results[-1]
print(f"Richer multi-beam representation (NOT matched budget): ~{rich_bits:.0f} bits, "
      f"match quality={rich_mq:.4f}, sum-rate={np.mean(rich_t2_rates):.3f} bits/s/Hz "
      f"(L={rich_cfg['L']}, amp_bits={rich_cfg['amp_bits']}, phase_bits={rich_cfg['phase_bits']})")

print("\nSaved experiment_4.png")
