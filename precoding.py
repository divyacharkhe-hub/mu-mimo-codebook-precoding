"""
DFT codebook-based MU-MIMO precoding over OFDM.

CONVENTION (fixed):
    Received signal for user k on a given subcarrier:
        y_k = h_k^H @ x + n_k ,   x = sum_i w_i s_i
    where h_k is the (Nt,) channel vector from the BS array to user k
    (stored, per subcarrier, as row k of H_used with NO conjugation --
    i.e. H_used[k, :] literally holds the channel taps).

    So the *effective* channel matrix every downstream computation must
    use is:
        G = H_used.conj() @ W          # (K, K)

    zf_precoder is therefore built to satisfy G ≈ I, i.e.
        H_used.conj() @ W ≈ I
    NOT H_used @ W ≈ I (that was the old, inconsistent derivation).

    Proof this holds: let A = H_used.conj()  (K x Nt, K <= Nt, full row
    rank a.s. for a random channel). np.linalg.pinv(A) = A^H (A A^H)^-1,
    and for full-row-rank A, A @ pinv(A) = I_K exactly. Setting
    W = pinv(H_used.conj()) therefore gives H_used.conj() @ W = I_K
    by construction -- no extra conjugate anywhere downstream.
"""

import numpy as np


# --------------------------------------------------------------------
# Channel model: geometric (ULA, clustered multipath) -- matches the
# angular structure a DFT/Type-I codebook is actually designed for.
# --------------------------------------------------------------------
def geometric_channel(Nt, n_paths, rng, aoa_spread_deg=5.0, d_over_lambda=0.5):
    """One user's channel vector (Nt,), sum of n_paths steering vectors
    with random complex gains and angles-of-arrival (degrees)."""
    h = np.zeros(Nt, dtype=complex)
    gains = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2)
    center_aoa = rng.uniform(-60, 60)
    aoas = center_aoa + rng.normal(0, aoa_spread_deg, n_paths)
    n = np.arange(Nt)
    for g, theta in zip(gains, aoas):
        steer = np.exp(1j * 2 * np.pi * d_over_lambda * n * np.sin(np.deg2rad(theta)))
        h += g * steer
    return h / np.sqrt(n_paths)


# --------------------------------------------------------------------
# DFT codebook + PMI selection
# --------------------------------------------------------------------
def dft_codebook(Nt, n_bits):
    """(2**n_bits, Nt) matrix of unit-norm DFT steering codewords."""
    L = 2 ** n_bits
    n = np.arange(Nt)
    # Oversampled DFT codebook: angles uniformly spaced in the virtual
    # angular domain [-1, 1).
    C = np.exp(1j * np.pi * np.outer(2 * np.arange(L) / L - 1, n))
    return C / np.sqrt(Nt)


def pmi_select(h, codebook):
    """Best codeword index + codeword for channel vector h (Nt,).
    Correct inner product: |h^H c|^2  ->  conj(h) dotted with c."""
    scores = np.abs(codebook @ h.conj()) ** 2
    idx = int(np.argmax(scores))
    return idx, codebook[idx]


# --------------------------------------------------------------------
# Zero-forcing precoder -- FIXED convention
# --------------------------------------------------------------------
def zf_precoder(H_used, total_power=1.0):
    """
    H_used : (K, Nt) complex, row k = h_k (unconjugated channel taps).
    Returns W : (Nt, K), column k = user k's precoding vector, power
    normalized so total transmit power == total_power.

    Guarantees H_used.conj() @ W ≈ I_K (up to the power-normalization
    scalar per column), matching the h_k^H w convention used by
    sum_rate / ber_sim.
    """
    K, Nt = H_used.shape
    assert K <= Nt, "zero-forcing requires K <= Nt"

    A = H_used.conj()                     # (K, Nt) -- the conjugated convention
    W = np.linalg.pinv(A)                 # (Nt, K); A @ W == I_K exactly

    # Equal power split across users, total power constraint.
    col_norms = np.linalg.norm(W, axis=0)         # (K,)
    W = W / col_norms                              # unit-norm columns
    W = W * np.sqrt(total_power / K)                # equal power split

    return W


# --------------------------------------------------------------------
# Type II-style (multi-beam) codebook: report L beams + quantized
# amplitude/phase combining coefficients, instead of a single beam.
# Simplified relative to 3GPP Rel-15 Type II (no subband/wideband
# split, no orthogonal-beam-group constraint) but structurally the
# same idea: h_hat = sum_l  a_l * e^{j phi_l} * beam_l.
# --------------------------------------------------------------------
def oversampled_beams(Nt, oversample=4):
    """(oversample*Nt, Nt) unit-norm DFT beams on an oversampled angular grid."""
    L = oversample * Nt
    n = np.arange(Nt)
    C = np.exp(1j * np.pi * np.outer(2 * np.arange(L) / L - 1, n))
    return C / np.sqrt(Nt)


def _quantize_amplitude(a, amp_bits):
    """Quantize a in [0,1] uniformly to 2**amp_bits levels."""
    levels = 2 ** amp_bits
    q = np.round(a * (levels - 1)) / (levels - 1)
    return q


def _quantize_phase(phi, phase_bits):
    """Quantize phase (radians) uniformly to 2**phase_bits levels."""
    levels = 2 ** phase_bits
    step = 2 * np.pi / levels
    return np.round(phi / step) * step


def type2_direction(h, beams, L, amp_bits=2, phase_bits=3):
    """
    Greedy (matching-pursuit) L-beam approximation of h. The strongest
    selected beam is used as a phase/amplitude reference: its relative
    phase is 0, while the remaining L-1 coefficients use quantized
    amplitudes and phases relative to that reference. This is Type-II-
    inspired rather than a full 3GPP Type II implementation.

    Returns h_hat (Nt,), unit-norm quantized channel direction estimate,
    and the feedback bit cost for this report.
    """
    n_beams = beams.shape[0]
    beam_idx_bits = int(np.ceil(np.log2(n_beams)))

    residual = h.copy()
    chosen = []
    coeffs = []
    for l in range(L):
        # proper projection coefficient: c_l = beam_l^H @ residual = conj(beam_l) . residual
        scores = beams.conj() @ residual
        idx = int(np.argmax(np.abs(scores)))
        c = scores[idx]
        chosen.append(idx)
        coeffs.append(c)
        residual = residual - c * beams[idx]      # subtract exactly the optimal projection

    coeffs = np.array(coeffs)
    ref = np.abs(coeffs[0])
    ref = ref if ref > 0 else 1.0
    ref_phase = np.angle(coeffs[0])

    h_hat = np.zeros_like(h)
    bits = L * beam_idx_bits
    for l, (idx, c) in enumerate(zip(chosen, coeffs)):
        if l == 0:
            # Global phase is irrelevant to the represented channel direction,
            # so the strongest beam defines phase zero.
            amp, phase = 1.0, 0.0
        else:
            amp = _quantize_amplitude(np.clip(np.abs(c) / ref, 0, 1), amp_bits)
            # Quantize phase RELATIVE to the reference beam. This avoids the
            # inconsistent old construction: reference phase=0, others absolute.
            rel_phase = np.angle(np.exp(1j * (np.angle(c) - ref_phase)))
            phase = _quantize_phase(rel_phase, phase_bits)
            bits += amp_bits + phase_bits
        h_hat += (ref * amp * np.exp(1j * phase)) * beams[idx]

    norm = np.linalg.norm(h_hat)
    if norm > 0:
        h_hat = h_hat / norm
    return h_hat, bits


def type2_precoder_inputs(H_true, beams, L, amp_bits=2, phase_bits=3):
    """H_hat matrix (K, Nt) of Type II-quantized directions + total bits/user."""
    K = H_true.shape[0]
    H_hat = np.zeros_like(H_true)
    bits_used = 0
    for k in range(K):
        h_hat, bits = type2_direction(H_true[k], beams, L, amp_bits, phase_bits)
        H_hat[k] = h_hat
        bits_used = bits   # same for every user given fixed L/amp_bits/phase_bits
    return H_hat, bits_used


# (codewords), not the true channel -- this is what actually happens
# with PMI feedback. Transmission still goes over the true channel,
# so residual interference shows up when evaluated against H_true.
# --------------------------------------------------------------------
def quantized_precoder(H_true, codebook, total_power=1.0):
    """
    H_true   : (K, Nt) true channel (unconjugated taps)
    codebook : (2**n_bits, Nt) unit-norm codewords
    Returns W (Nt, K) built from each user's best-match codeword,
    and the list of PMI indices selected.
    """
    K, Nt = H_true.shape
    pmis = []
    H_hat = np.zeros((K, Nt), dtype=complex)
    for k in range(K):
        idx, cw = pmi_select(H_true[k], codebook)
        pmis.append(idx)
        H_hat[k] = cw
    W = zf_precoder(H_hat, total_power=total_power)
    return W, pmis



def sum_rate(H, W, noise_power):
    """
    H : (K, Nt) channel (unconjugated taps, row k = h_k)
    W : (Nt, K) precoder, column k = w_k
    Returns total sum rate (bits/s/Hz) over K users.
    """
    G = H.conj() @ W                      # (K, K), same convention as zf_precoder
    K = G.shape[0]
    powers = np.abs(G) ** 2               # |h_j^H w_k|^2  at [j,k]

    rate = 0.0
    for k in range(K):
        signal = powers[k, k]
        interf = powers[k, :].sum() - signal
        sinr = signal / (interf + noise_power)
        rate += np.log2(1 + sinr)
    return rate


# --------------------------------------------------------------------
# QPSK Monte Carlo BER
# --------------------------------------------------------------------
def _qpsk_mod(bits):
    # bits: (..., 2) -> complex symbol, Gray-mapped, unit average power
    b0, b1 = bits[..., 0], bits[..., 1]
    real = 1 - 2 * b0
    imag = 1 - 2 * b1
    return (real + 1j * imag) / np.sqrt(2)


def _qpsk_demod(sym):
    b0 = (sym.real < 0).astype(int)
    b1 = (sym.imag < 0).astype(int)
    return np.stack([b0, b1], axis=-1)


def ber_sim(H, W, snr_db, n_symbols, rng):
    """
    Monte Carlo BER for K users sharing the precoded MU-MIMO channel.
    H : (K, Nt) unconjugated channel taps
    W : (Nt, K) precoder from zf_precoder (same H!)
    """
    K = H.shape[0]
    G = H.conj() @ W                      # (K, K) -- consistent convention

    noise_power = 10 ** (-snr_db / 10)

    bits_tx = rng.integers(0, 2, size=(n_symbols, K, 2))
    s = _qpsk_mod(bits_tx)                # (n_symbols, K)

    n = (rng.standard_normal((n_symbols, K)) + 1j * rng.standard_normal((n_symbols, K)))
    n *= np.sqrt(noise_power / 2)

    y = s @ G.T + n                       # y_k = sum_i G[k,i] s_i + n_k

    # Zero-forcing already nulls interference (G ≈ I up to power scale),
    # so a simple per-user rescale by the diagonal recovers the symbol.
    diag = np.diag(G)
    y_eq = y / diag                       # (n_symbols, K)

    bits_rx = _qpsk_demod(y_eq)
    ber = np.mean(bits_tx != bits_rx)
    return ber


# --------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Nt, K = 4, 2          # K < Nt: spare degrees of freedom
    n_paths = 3

    H = np.stack([geometric_channel(Nt, n_paths, rng) for _ in range(K)])
    W = zf_precoder(H, total_power=1.0)

    G = H.conj() @ W
    print("H.conj() @ W  (should be ~ (1/K, 0; 0, 1/K)-ish, diag >> off-diag):")
    print(np.round(G, 3))

    print("\nBER at high SNR (should -> 0 for perfect-CSI, K<Nt ZF):")
    for snr in [0, 10, 20, 30]:
        ber = ber_sim(H, W, snr, n_symbols=20000, rng=rng)
        print(f"  SNR={snr:>3} dB   BER={ber:.5f}")
