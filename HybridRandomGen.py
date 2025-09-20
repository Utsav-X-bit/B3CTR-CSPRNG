from HybridRng import HybridRNG
import time

hybrid_rng = HybridRNG()
with open ("rng_output.txt" , 'w') as f:
        
    start = time.perf_counter()
    for i in range(10000):
            a = f.write(str(hybrid_rng.next_int(1,99))+"\n")
            print(a)
    end = time.perf_counter()
    print(f" Hybrid Rng took {end - start:.3f} seconds")