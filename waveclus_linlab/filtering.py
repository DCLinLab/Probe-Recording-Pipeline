from scipy.signal import sosfiltfilt, butter, correlate
import numpy as np


def bandpass(data_all, fs, pass_band=(250, 6000), order=4, iter=1):
    """
    channel by channel bandpass filtering

    :param data_all: multichannel ephys to process
    :param fs: sampling frequency
    :param pass_band: (low, high) bandpass
    :param order: order of butterworth
    :param iter: # of iterations
    :return:
    """
    nq = .5 * fs
    low = pass_band[0] / nq
    high = pass_band[1] / nq
    sos = butter(order, [low, high], 'band', output='sos')
    for i in range(iter):
        data_all = sosfiltfilt(sos, data_all)
    return data_all


def clean_mech_noise(data_all, mov_window=2000, interval=100):
    """
    Considering all the channal to remove mechanical noise, from Guosong

    :param data_all: multichannel ephys to process
    :param mov_window:
    :param interval:
    :return:
    """
    # Determine the dimension to average over
    reference_data = np.mean(data_all, axis=0)

    # Initialize the array to hold the moving cross-correlation noise
    mov_ccnoise = np.zeros(len(reference_data))

    # Compute the moving cross-correlation noise
    for i in range(0, len(reference_data) - interval + 1, interval):
        start_idx = max(i - mov_window // 2, 0)
        end_idx = min(i + mov_window // 2, len(reference_data))
        idx = np.arange(start_idx, end_idx)
        if len(idx) > 0:
            # Compute the cross-correlation
            cc = correlate(reference_data[idx], reference_data[idx], mode='full')
            mov_ccnoise[i:i + interval] = np.max(cc)

    # Normalize the cross-correlation noise
    mov_ccnoise = (mov_ccnoise - np.min(mov_ccnoise)) / (np.max(mov_ccnoise) - np.min(mov_ccnoise))
    idx_clean = np.where(mov_ccnoise < np.median(mov_ccnoise) * 2.5)[0]

    # Initialize the cleaned data array
    data_all_cleaned = np.zeros_like(data_all)

    # Fill the cleaned data based on the dimension to average over
    data_all_cleaned[:, idx_clean] = data_all[:, idx_clean]

    return data_all_cleaned, idx_clean