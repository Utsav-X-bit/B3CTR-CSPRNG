# HybridRng.py
import os
import time
import math
import random
import argparse
import numpy as np
from blake3 import blake3
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Hybrid RNG Implementation (compatible with NIST test suite)
# =========================================================
import os, time
from blake3 import blake3

class HybridRNG:
    def __init__(self, seed=None, reseed_interval=0, verbose=True):
        # Accept int|bytes|str|None
        if seed is None:
            seed_int = int.from_bytes(os.urandom(32), "big") ^ time.time_ns()
            seed_bytes = seed_int.to_bytes(32, "big", signed=False)
        elif isinstance(seed, int):
            seed_bytes = seed.to_bytes(32, "big", signed=False)
        elif isinstance(seed, str):
            seed_bytes = seed.encode()
        elif isinstance(seed, (bytes, bytearray)):
            seed_bytes = bytes(seed)
        else:
            raise TypeError("seed must be int|bytes|str|None")

        # Derive a 32-byte secret key from the provided seed
        self.secret_key = blake3(seed_bytes).digest()
        self.state = blake3(b"state:" + self.secret_key).digest()
        self.counter = 0
        self.reseed_interval = reseed_interval

        # Buffer for efficiency
        self._buffer = b""
        self._buffer_pos = 0

        if verbose:
            print(f"[INIT] seed_repr: {int.from_bytes(seed_bytes[-8:], 'big')} (last 8 bytes as int)")
            print(f"[INIT] state64 (hex): {self.state[:8].hex()}")

    def reseed(self, extra_input=None):
        """Reseed by mixing new entropy into the secret key and state."""
        if extra_input is None:
            extra = os.urandom(16) + time.time_ns().to_bytes(8, "big")
        elif isinstance(extra_input, int):
            extra = extra_input.to_bytes(32, "big", signed=False)
        elif isinstance(extra_input, str):
            extra = extra_input.encode()
        elif isinstance(extra_input, (bytes, bytearray)):
            extra = bytes(extra_input)
        else:
            raise TypeError("extra_input must be int|bytes|str|None")

        self.secret_key = blake3(self.secret_key + extra).digest()
        self.state = blake3(self.state + extra).digest()

    def _refill_buffer(self):
        """Refill internal buffer with a new 256-bit block."""
        self.counter += 1
        if self.reseed_interval and self.counter % self.reseed_interval == 0:
            self.reseed()
        counter_bytes = self.counter.to_bytes(32, "big")
        self._buffer = blake3(counter_bytes, key=self.secret_key).digest()
        self._buffer_pos = 0

    def _consume(self, nbytes):
        """Consume nbytes from buffer (refill if needed)."""
        out = bytearray()
        while nbytes > 0:
            if self._buffer_pos >= len(self._buffer):
                self._refill_buffer()
            chunk = min(nbytes, len(self._buffer) - self._buffer_pos)
            out.extend(self._buffer[self._buffer_pos:self._buffer_pos+chunk])
            self._buffer_pos += chunk
            nbytes -= chunk
        return bytes(out)

    # ======================== Random output APIs ========================

    def rand_raw(self):
        """Return a 64-bit unsigned random integer."""
        return int.from_bytes(self._consume(8), "big")

    def next(self):
        """Uniform float in [0,1)."""
        val = int.from_bytes(self._consume(32), "big")
        return val / (1 << 256)

    def next_in_range(self, a=0.0, b=1.0):
        return a + self.next() * (b - a)

    def next_uint(self, bits: int = 64):
        if not (1 <= bits <= 256):
            raise ValueError("bits must be in [1, 256]")
        val = int.from_bytes(self._consume(32), "big")
        return val >> (256 - bits)

    def next_int(self, min_val: int, max_val: int):
        if max_val < min_val:
            raise ValueError("max_val must be >= min_val")
        span = max_val - min_val + 1
        limit = ((1 << 256) // span) * span
        while True:
            r = int.from_bytes(self._consume(32), "big")
            if r < limit:
                return min_val + (r % span)

    def next_bytes(self, n):
        """Efficiently get n random bytes (needed for NIST tests)."""
        return self._consume(n)


# =========================================================
# Simple Statistical Tests (for quick checks)
# =========================================================
def chi_square_test(samples, n_bins=100):
    counts, _ = np.histogram(samples, bins=n_bins, range=(0,1))
    expected = len(samples) / n_bins
    chi2 = ((counts - expected)**2 / expected).sum()
    df = n_bins - 1
    p_value = chi2_dist.sf(chi2, df)
    return chi2, p_value

def shannon_entropy(samples, n_bins=100):
    counts, _ = np.histogram(samples, bins=n_bins, range=(0,1))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def runs_test(samples):
    median = np.median(samples)
    runs, n1, n2 = 0, 0, 0
    prev = None
    for s in samples:
        curr = s >= median
        if curr != prev:
            runs += 1
        prev = curr
        if curr:
            n1 += 1
        else:
            n2 += 1
    if n1 == 0 or n2 == 0:
        return runs, 0
    expected = 1 + (2*n1*n2)/(n1+n2)
    var = (2*n1*n2*(2*n1*n2 - n1 - n2)) / (((n1+n2)**2)*(n1+n2-1))
    z = (runs - expected)/math.sqrt(var)
    return runs, z

def autocorr(samples, lag=1):
    n = len(samples)
    mean = np.mean(samples)
    num = np.sum((samples[:n-lag] - mean) * (samples[lag:] - mean))
    den = np.sum((samples - mean)**2)
    return num/den if den > 0 else 0

def plot_histograms(hybrid_samples, py_samples, np_samples, n_bins=50, save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)

    rng_data = {
        "Hybrid_RNG": hybrid_samples,
        "Python_RNG": py_samples,
        "NumPy_RNG": np_samples,
    }

    for name, samples in rng_data.items():
        plt.figure(figsize=(8, 5))
        plt.hist(samples, bins=n_bins, range=(0, 1), alpha=0.7, label=name)
        plt.legend()
        plt.title(f"Histogram of {name}")
        save_path = os.path.join(save_dir, f"hist_{name}.png")
        plt.savefig(save_path)
        print(f"[PLOT] Saved {name} histogram to {save_path}")
        plt.close()


# =========================================================
# Runner
# =========================================================
def evaluate_rng(samples, name, n_bins=100):
    chi2, p = chi_square_test(samples, n_bins)
    entropy = shannon_entropy(samples, n_bins)
    runs, z = runs_test(samples)
    ac = autocorr(samples, lag=1)

    print(f"\n----- {name} -----")
    print(f"Chi-Square: χ²={chi2:.2f}, p={p:.4f} -> {'PASS' if p>0.05 else 'FAIL'}")
    print(f"Shannon Entropy: {entropy:.6f} bits (ideal ~{math.log2(n_bins):.4f})")
    print(f"Runs Test: runs={runs}, z={z:.3f} -> {'PASS' if abs(z)<1.96 else 'FAIL'}")
    print(f"Autocorrelation (lag=1): {ac:.6f} -> {'PASS' if abs(ac)<0.05 else 'FAIL'}")


def run_all(n_samples=10000, n_bins=100, reseed_interval=0, seed_for_hybrid=None):
    print("=== Hybrid RNG Quick Evaluation ===")
    rng = HybridRNG(seed_for_hybrid, reseed_interval=reseed_interval)

    print("\nGenerating samples...")
    hybrid_samples = [rng.next() for _ in range(n_samples)]

    # Save bitstream for external tests
    bit_stream = ''.join('1' if x > 0.5 else '0' for x in hybrid_samples)
    with open("rng_bit_output.txt", "w") as f:
        f.write(bit_stream)

    with open("rng_output.txt", "w") as f:
        for x in hybrid_samples:
            f.write('1\n' if x > 0.5 else '0\n')

    # Compare with Python and NumPy RNGs
    py_samples = [random.random() for _ in range(n_samples)]
    np_samples = np.random.random(n_samples)

    evaluate_rng(hybrid_samples, "Hybrid RNG", n_bins)
    evaluate_rng(py_samples, "Python RNG", n_bins)
    evaluate_rng(np_samples, "NumPy RNG", n_bins)

    plot_histograms(hybrid_samples, py_samples, np_samples, n_bins)

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RNG Runner & Quick Tests")
    parser.add_argument("--samples", type=int, default=10000, help="Number of random samples to generate")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for chi-square and entropy tests")
    parser.add_argument("--reseed_interval", type=int, default=0, help="Reseed interval (0 disables reseeding)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    args = parser.parse_args()

    run_all(n_samples=args.samples, n_bins=args.bins,
            reseed_interval=args.reseed_interval, seed_for_hybrid=args.seed)
