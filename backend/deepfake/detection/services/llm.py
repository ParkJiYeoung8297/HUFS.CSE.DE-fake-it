import time


def measure_llm_generation(func, timings, *args, **kwargs):
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    timings['llm'] = time.perf_counter() - start_time
    
    return result