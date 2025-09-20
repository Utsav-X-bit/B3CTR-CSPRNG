import time
import random
import numpy as np
from blake3 import blake3
import os
import struct

MASK64 = (1 << 64) - 1

def rotl(x: int, r: int) -> int:
    r &= 63
    return ((x << r) & MASK64) | (x >> (64 - r))

class B3GUARD:
    """
    Counter-mode BLAKE3 with key derived from seed. Each 32-byte block is
    mixed into a 64-bit output using cross-chunk whitening + SplitMix64.
    Optional periodic reseeding can be enabled for extremely long runs.
    """

    def __init__(self, seed: int = None, burn_in: int = 0, reseed_interval: int | None = None):
        """
        seed: integer seed (None -> from os.urandom, but then it's non-reproducible)
        burn_in: number of initial outputs to discard
        reseed_interval: if set (e.g., 1_000_000), periodically reseeds the secret key
                         using a hash of the current key and fresh entropy. None -> off.
        """
        if seed is None:
            # Still allow deterministic behavior if user passes a seed explicitly
            seed = int.from_bytes(os.urandom(32)) ^ time.time_ns()
            seed_bytes = seed.to_bytes(32,'big', signed=False)
        else:
            seed_bytes = seed.to_bytes(32, 'big', signed=False)

        # Initial key from BLAKE3 of the seed bytes
        self._secret_key = blake3(seed_bytes).digest()
        self._counter = 0
        self._extract_mode = 0  # cycles 0..3 to vary intra-block mixing
        self._reseed_interval = reseed_interval

        # Optional burn-in to decorrelate initial state
        for _ in range(burn_in):
            _ = self.rand_raw()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _next_block(self) -> bytes:
        # Hash key || counter to produce a 32-byte block, then advance state
        ctr_bytes = self._counter.to_bytes(16, 'big', signed=False)
        hasher = blake3(self._secret_key)
        hasher.update(ctr_bytes)
        block = hasher.digest()  # 32 bytes
        self._counter += 1

        # Lightweight key update to prevent static-key structure in very long streams
        # (Keeps determinism; not adding external entropy here)
        self._secret_key = blake3(block + self._secret_key).digest()
        self._secret_key = blake3(self._secret_key).digest()   # double-mix

        # Optional periodic reseed (disabled by default to preserve reproducibility)
        if self._reseed_interval and (self._counter % self._reseed_interval == 0):
            self._secret_key = blake3(self._secret_key + os.urandom(32)).digest()

        return block

    @staticmethod
    def _splitmix64(x: int) -> int:
        # Standard SplitMix64 finalizer
        x = (x + 0x9E3779B97F4A7C15) & MASK64
        x ^= (x >> 30)
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= (x >> 27)
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= (x >> 31)
        return x & MASK64

    def rand_raw(self) -> int:
        """
        Produce a 64-bit unsigned integer with whitening and SplitMix64 finalization.
        The mixing uses all four 64-bit lanes of the 32-byte BLAKE3 block,
        and cycles the rotation pattern each call to reduce structural artifacts.
        """
        block = self._next_block()
        # Interpret 32-byte block as 4x uint64
        p0, p1, p2, p3 = struct.unpack('>QQQQ', block)

        mode = self._extract_mode
        self._extract_mode = (self._extract_mode + 1) & 3

        # Cross-lane mixing with rotating schedule per mode
        if mode == 0:
            mixed = p0 ^ rotl(p1, 13) ^ rotl(p2, 29) ^ rotl(p3, 47)
        elif mode == 1:
            mixed = p1 ^ rotl(p2, 17) ^ rotl(p3, 41) ^ rotl(p0, 7)
        elif mode == 2:
            mixed = p2 ^ rotl(p3, 23) ^ rotl(p0, 31) ^ rotl(p1, 11)
        else:  # mode == 3
            mixed = p3 ^ rotl(p0, 19) ^ rotl(p1, 37) ^ rotl(p2, 5)

        # Final scrambler to kill residual linear structure (helps Linear Complexity)
        return self._splitmix64(mixed)

    def next_uint64(self) -> int:
        return self.rand_raw()

    def next(self) -> float:
        # Convert to float in [0,1) with 53-bit mantissa precision
        x = self.rand_raw()
        # Take top 53 bits for uniform double conversion
        return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    # Convenience: stream of bits for NIST tests
    def bits(self, n_bits: int) -> str:
        out = []
        while len(out) * 64 < n_bits:
            v = self.rand_raw()
            out.append(f'{v:064b}')
        bitstring = ''.join(out)
        return bitstring[:n_bits]

def benchmark(n: int = 10_000_000):
    """
    Benchmark B3GUARD, Python's random, and NumPy random
    for generating n random floats in [0,1).
    """
    from __main__ import B3GUARD   # if B3GUARD is in same file

    rng = B3GUARD(seed=None)

    results = {}

    # --- B3GUARD ---
    start = time.perf_counter()
    for _ in range(n):
        rng.next()
    results["B3GUARD"] = time.perf_counter() - start

    # --- Python random ---
    start = time.perf_counter()
    for _ in range(n):
        random.random()
    results["Python random"] = time.perf_counter() - start

    # --- NumPy random ---
    start = time.perf_counter()
    np.random.random(n)   # vectorized
    results["NumPy random"] = time.perf_counter() - start

    # Print results
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f} seconds")


if __name__ == "__main__":
    benchmark(10_000_0)  # try 1e6 first, then go to 1e7
