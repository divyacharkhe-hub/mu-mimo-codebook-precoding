# Codebook-Based MU-MIMO Precoding: What Limited Feedback Actually Costs You

My last post assumed something MU-MIMO systems don't actually have: perfect
CSI. Every user reporting their exact channel back to the base station, every
bit of it. Real systems can't afford that — so instead users report a
*quantized* version of their channel direction, a few bits picked from a
codebook, and the base station builds its zero-forcing precoder from that
instead.

This post is about what that quantization actually costs you, and about several
bugs I had to find before I could trust any of the numbers below.

## The setup

DFT codebook, MU-MIMO downlink, OFDM. Each user picks the codeword that best
matches its channel direction and reports its index — a simplified PMI-style
feedback model. The base station collects those indices, builds a ZF precoder
from the quantized directions, and transmits.

I wanted to answer two questions: how much sum-rate do we lose to limited
feedback, and does a richer **multi-beam, Type-II-inspired** representation win
back some of the loss that a single beam structurally cannot?

One terminology fix matters here: my channel generator uses one random angular
center with several nearby propagation paths. With `n_paths=3`, this is a
**3-path clustered multipath channel**, not three independent clusters.

## Bug #1: the PMI selection itself

First thing I found, before any of the real experiments: perfect CSI was
scoring *worse* than the quantized codebook. Backwards. The PMI match score has
to use the Hermitian inner product `|h^H c|²`. I had the conjugation flipped.
Small mistake, but it silently contaminates every number downstream.

## Bug #2: the channel model didn't match the codebook

Once PMI selection was fixed, the match-quality curves still looked wrong. I
was testing a DFT steering-vector codebook against plain i.i.d. Rayleigh
vectors with no angular structure.

That is not the experiment I thought I was running.

I switched to a geometric ULA channel where the taps are built from physical
steering vectors around a random arrival angle. Then I made one more cleanup:
every feedback-bit value is now evaluated on the **same 400 channel draws**.
Because the DFT codebooks are nested, that removes irrelevant Monte Carlo
jitter and makes the match-quality trend monotonic.

The result:

| bits | single-path | 3-path clustered multipath |
|---:|---:|---:|
| 1 | 0.3846 | 0.3891 |
| 2 | 0.7532 | 0.7219 |
| 3 | 0.9404 | 0.8997 |
| 5 | 0.9961 | 0.9513 |
| 10 | 1.0000 | 0.9551 |

The single-path channel converges to essentially perfect directional match.
The 3-path channel does not. It plateaus around 0.955.

That is the real limitation: a rank-1 beam can resolve **which single beam**
better and better, but it still cannot represent a channel whose energy lives
across several angular components.

More index bits do not fix a representation-rank problem.

## Bug #3: the conjugate convention

This was the expensive one.

Perfect-CSI BER was not going to zero at high SNR in a case where zero-forcing
mathematically had enough spatial degrees of freedom to null interference. I
isolated the problem to a convention mismatch.

The receiver model in the simulation is

```text
y_k = h_k^H x + n_k
```

so the effective MU-MIMO channel is

```python
G = H.conj() @ W
```

But the original ZF construction was effectively solving a different identity
based on the unconjugated channel. The code called both things "H" and the bug
looked like residual interference.

The fix was to build the precoder from the same convention the receiver uses:

```python
W = np.linalg.pinv(H.conj())
```

before the per-column power normalization. Now perfect-CSI ZF gives a diagonal
effective channel, exactly as it should.

The "interference floor" I was chasing there was not physics. It was
bookkeeping.

## What limited feedback actually costs

With the convention fixed, I moved to the continuous metric that actually
shows the damage clearly: sum-rate.

For a fully loaded system (`K = Nt = 4`) at 20 dB, using the same 300 channel
matrices at every feedback level:

| bits/user | mean sum-rate (bits/s/Hz) |
|---:|---:|
| 1 | 2.830 |
| 2 | 5.633 |
| 3 | 8.176 |
| 4 | 9.199 |
| 5 | 9.535 |
| 6 | 9.496 |
| 7 | 9.527 |
| 8 | 9.451 |
| 9 | 9.403 |
| 10 | 9.390 |
| perfect CSI | 11.665 |

The useful gain comes fast. Going from one to roughly five feedback bits makes
a huge difference. After that, the system sits around a 9.4–9.5 bits/s/Hz
region while perfect CSI is at 11.665.

Notice that the sum-rate curve is not perfectly monotonic after five bits even
though the per-user match-quality curve is. That is not a contradiction. A
fully loaded ZF precoder depends on the **joint conditioning of all four
quantized user directions**. A slightly better individual direction match can
still produce a slightly worse multi-user geometry for one channel draw.

The important point is the persistent gap: better single-beam angular
resolution does not remove the underlying multi-path representation error.

## The metric trap: BER tells a different story

I originally wanted BER to show a smooth high-SNR floor that gradually closed
as feedback bits increased.

It did not.

For the fixed, well-conditioned channel used in the BER experiment, the 3-bit
and 6-bit PMI curves follow normal waterfall behavior and eventually reach zero
observed errors in the Monte Carlo run. The dramatic case is 1-bit feedback:
two users choose the same PMI, the quantized channel matrix becomes rank
deficient, and ZF cannot separate all users anymore.

Its aggregate BER approaches roughly 0.25 at high SNR.

**0.25 is not "random guessing." Fully random bit guessing would be BER=0.5.**
Here ~0.25 is an aggregate result from this specific partial-collapse geometry —
some user separation survives and some does not.

So BER is useful, but it tells a sharper threshold story:

- moderate quantization → SNR penalty / shifted waterfall;
- severe PMI collision → rank deficiency and catastrophic separation failure.

For the smoother "how much performance does quantization continuously cost?"
question, sum-rate is the better metric.

## Does a multi-beam representation actually help?

Yes — but I wanted the comparison to be fair.

For a fair feedback-budget comparison, I use the 13-bit `L=2` multi-beam
configuration against a 13-bit single-beam codebook. The richer 26-bit
multi-beam configuration is reported separately as an unequal-budget reference.

For the phase representation, the strongest selected beam defines phase zero
and every other coefficient is quantized **relative to that reference**. Global
phase does not affect the represented channel direction, while relative phase
controls how the selected beams combine.

This is still a **Type-II-inspired model**, not a full 3GPP Type II
implementation.

Match quality:

| configuration | bits/user | match quality |
|---|---:|---:|
| Type I single beam | 13 | 0.9572 |
| multi-beam, L=2 | 13 | 0.9774 |
| multi-beam, L=3 | 22 | 0.9812 |
| multi-beam, L=3 with richer coefficients | 26 | 0.9945 |

And at the actually matched 13-bit budget:

| scheme | mean sum-rate (bits/s/Hz) |
|---|---:|
| Type I, 13 bits | 9.072 |
| multi-beam L=2, 13 bits | 9.467 |
| perfect CSI | 11.529 |

So even at equal feedback cost, representing the channel with two beams gives
a measurable improvement over spending all 13 bits resolving one beam more
finely.

If I allow the richer 26-bit multi-beam configuration, it reaches 0.9945 match
quality and 10.599 bits/s/Hz. That is a useful representation result, but it is
**not an equal-feedback comparison** and I am keeping it labeled that way.

That distinction matters more to me than making the headline number look
bigger.

## What I actually learned

The interesting part of this project ended up being less "more feedback bits
are better" and more about **what kind of error the feedback representation
can express**.

A single-beam codebook has two different limits:

1. **resolution error** — too few bits to choose the right beam accurately;
2. **representation error** — one beam is the wrong model for a channel with
   multiple angular components.

More Type-I bits attack the first problem. They do almost nothing to the
second.

That is why the single-path curve goes to 1.0 while the 3-path clustered curve
stops around 0.955, and why a two-beam representation can beat a much finer
single-beam codebook at the same total feedback budget.

The debugging also changed how I read simulation plots. A BER floor can be a
channel effect, a conditioning effect, a rank-collapse effect — or just a
conjugate in the wrong place. The curve itself does not tell you which one.
You have to isolate the mechanism.

## Next

Subband vs wideband PMI reporting is the natural follow-on. Everything above
uses one channel direction report per user, while a real OFDM channel is
frequency selective. The next question is whether spending feedback across
subbands is more valuable than spending the same budget on a richer spatial
representation.
