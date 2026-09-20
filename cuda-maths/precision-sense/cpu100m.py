# cpu100m.py - the CPU vectorised path, 12 workers below normal, own process so the
# pool's re-import of __main__ cannot touch the GPU script.
import sys, json, time
sys.path.insert(0, r"(local path removed)")
if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    import cpu_vec
    n = int(sys.argv[1])
    t0 = time.time()
    counts, _ = cpu_vec.run_multiproc(n, workers=12)
    print(json.dumps({"counts": [int(x) for x in counts], "seconds": time.time() - t0}))
