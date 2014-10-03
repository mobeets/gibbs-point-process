import numpy as np

def rolling_window(a, window, skip=1):
    """
    a is np.array
    window is int, the number of samples in each window
    skip is int; if skip==1 (default) this returns every window
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::skip]

def spike_stats(xs, window_size, skip=1):
    bin_counts = np.sum(rolling_window(xs, window_size), -1).flatten()
    return np.mean(bin_counts), np.var(bin_counts) 
