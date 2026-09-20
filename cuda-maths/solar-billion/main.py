# main.py - the entry point. The guard is required on Windows: the worker processes
# re-import the main module, and without it they would re-run the whole job.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import run_all   # noqa: F401  (running the module IS the job)
