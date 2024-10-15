# spike_detection.py
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.interpolate import splrep, splev
import spikeinterface as si


def process_channel(r_detect, r_sort, seg_dur, stdmin, stdmax, min_ref_per, detect, w_pre, w_post, fct):
    tot_dur = r_detect.get_duration()
    nseg = math.ceil(tot_dur / seg_dur)
    fs = r_detect.get_sampling_frequency()
    ref = int(min_ref_per * fs / 1000)
    sample_ref = np.floor(ref / 2)
    gain = r_sort.get_channel_gains()[0]
    offset = r_sort.get_channel_offsets()[0]
    ls = w_pre + w_post

    if detect == 'neg':
        func = lambda x: -x
    elif detect == 'pos':
        func = lambda x: +x
    elif detect == 'both':
        func = lambda x: np.abs(x)
    else:
        raise ValueError(f"Invalid value {detect} for argument 'detect'. Must be 'neg', 'pos', or 'both'.")

    times, waveforms = [], []
    # process by segments for memory efficiency
    for seg_ind in range(nseg):
        start_time = seg_ind * seg_dur
        end_time = min((seg_ind + 1) * seg_dur, tot_dur)

        trace_bp4 = r_detect.get_traces(start_frame=int(start_time * fs), end_frame=int(end_time * fs))
        trace_bp2 = r_sort.get_traces(start_frame=int(start_time * fs), end_frame=int(end_time * fs))

        thr = stdmin * np.median(np.abs(trace_bp4)) / 0.6745        # lower limit
        thrmax = stdmax * np.median(np.abs(trace_bp2)) / 0.6745     # upper limit
        xaux = np.where(func(trace_bp4[w_pre + 2: -w_post - 2 - int(sample_ref)]) > thr)[0] + w_pre + 1

        xaux0 = 0
        index = []
        for a in xaux:
            if a >= xaux0 + ref:
                # Check and eliminate artifacts
                tip = func(trace_bp2[a: a + int(sample_ref)])
                iaux = np.argmax(tip)
                if tip[iaux] > thrmax:
                    continue
                if iaux == 0 and func(trace_bp2[a + 1: a + int(sample_ref) + 1]).max() < thr:
                    continue
                index.append(iaux + a)
                xaux0 = index[-1]
        times.extend((np.array(index) / fs + start_time) * 1000)

        # waveform extraction
        for i in index:
            ind = np.clip(i + np.arange(-w_pre, w_post), 0, len(trace_bp2) - 1)
            tck = splrep(np.arange(ls), trace_bp2[ind] * gain + offset)
            interp = splev(np.arange(0, ls, 1 / fct), tck)
            p1, p2 = int((w_pre - 1) * fct), int((w_pre + 1) * fct)
            iaux = int(func(interp[p1:p2]).argmax() - fct + 1)
            ind = np.clip([iaux + int(i * fct) for i in range(ls)], 0, len(interp) - 1)
            waveforms.append(interp[ind])

    return {'times': np.array(times) , 'waveforms': np.vstack(waveforms),
            'fs': fs, 'dur': tot_dur, 'noise': si.get_noise_levels(r_sort, return_scaled=True)[0]}


def detect_spikes(recording, r_sort, r_detect, max_workers, **config):
    """Detects spikes from a given recording and all channels.

    Args:
        recording: The recording object to process.
        detect: The detection method ('neg', 'pos', 'both'). Default is 'neg'.
        segment_duration: Duration of each segment in seconds. Default is 5 minutes.
        stdmin: Threshold factor for detection. Default is 5.
        stdmax: Threshold factor for artifact removal. Default is 50.
        w_pre: Pre-event window size.
        w_post: Post-event window size.
        min_ref_per: Minimum refractory period in milliseconds. Default is 1.5.

    Returns:
        results: A dictionary where the keys are the channel ids, and the values are
                  another dictionary with the keys 'spikes', 'thresholds' and 'indices'.
    """
    detect = config.get('detect_method', 'both')
    seg_dur = config.get('segment_duration', 5) * 60
    min_ref_per = config.get('min_ref_per', 1.5)
    w_pre = config.get('w_pre', 20)
    w_post = config.get('w_post', 44)
    std_min = config.get('std_min', 5)
    std_max = config.get('std_max', 50)
    int_factor = config.get('int_factor', 5)

    with ProcessPoolExecutor(max_workers) as executor:
        # Submit tasks for each channel to the ThreadPoolExecutor
        futures, results = [], []
        for ch in recording.channel_ids:
            futures.append(
                executor.submit(process_channel,
                                r_detect.select_channels([ch]), r_sort.select_channels([ch]),
                                seg_dur, std_min, std_max, min_ref_per, detect, w_pre, w_post, int_factor)
            )
        # Collect the results for each channel as they become available
        for fut in tqdm(futures, 'Spike Detection', unit='channel'):
            results.append(fut.result())

    return dict(zip(recording.channel_ids, results))
