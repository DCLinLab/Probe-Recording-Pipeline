# spike_detection.py
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import spikeinterface as si


def process_channel(sub_recording_bp4, sub_recording_bp2, num_segments, segment_duration,
                    total_duration, sr, stdmin, stdmax, detect, w_pre, w_post, sample_ref, ref, merge_radius):

    all_spikes = []
    indices = []

    if detect == 'neg':
        func = lambda x: -x
    elif detect == 'pos':
        func = lambda x: +x
    elif detect == 'both':
        func = lambda x: np.abs(x)
    else:
        raise ValueError(f"Invalid value {detect} for argument 'detect'. Must be 'neg', 'pos', or 'both'.")

    for segment_index in range(num_segments):
        start_time = segment_index * segment_duration
        end_time = min((segment_index + 1) * segment_duration, total_duration)

        trace_bp4 = sub_recording_bp4.get_traces(start_frame=int(start_time * sr), end_frame=int(end_time * sr))
        trace_bp2 = sub_recording_bp2.get_traces(start_frame=int(start_time * sr), end_frame=int(end_time * sr))

        thr = stdmin * np.median(np.abs(trace_bp4)) / 0.6745
        thrmax = stdmax * np.median(np.abs(trace_bp2)) / 0.6745
        xaux = np.where(func(trace_bp4[w_pre + 2: -w_post - 2 - int(sample_ref)]) > thr)[0] + w_pre + 1

        # xaux2 = xaux.copy()
        # for i in range(len(xaux)):
        #     j = 1
        #     mid = xaux[i]
        #     while i - j >= 0 and mid - xaux[i - j] <= merge_radius:
        #         if func(trace_bp4[xaux[i - j]]) > func(trace_bp4[xaux[i]]):
        #             xaux2[i] = xaux[i - j]
        #         j += 1
        #     j = 1
        #     while i + j < len(xaux) and xaux[i + j] - mid <= merge_radius:
        #         if func(trace_bp4[xaux[i + j]]) > func(trace_bp4[xaux[i]]):
        #             xaux2[i] = xaux[i + j]
        #         j += 1
        #
        # xaux = np.unique(xaux2)

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

        spike_times = (np.array(index) / sr + start_time) * 1000

        all_spikes.extend(spike_times)
        indices.extend(index)
    return {'spikes_time': np.array(all_spikes), 'spikes_index': np.array(indices),
            'noise': si.get_noise_levels(sub_recording_bp2, return_scaled=True)[0]}


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
    detect = config['detect_method']
    segment_duration = config['segment_duration'] * 60
    min_ref_per = config['min_ref_per']
    
    total_duration = r_detect.get_num_frames() / r_detect.get_sampling_frequency()
    num_segments = math.ceil(total_duration / segment_duration)

    sr = r_detect.get_sampling_frequency()
    ref = int(min_ref_per * sr / 1000)
    sample_ref = np.floor(ref/2)
    results = {}

    with ProcessPoolExecutor(max_workers) as executor:
        # Submit tasks for each channel to the ThreadPoolExecutor
        futures = []
        for channel_id in recording.channel_ids:
            futures.append(
                executor.submit(process_channel,
                                r_detect.select_channels([channel_id]), r_sort.select_channels([channel_id]),
                                num_segments, segment_duration, total_duration, sr, config['std_min'], config['std_max'],
                                detect, config['w_pre'], config['w_post'], sample_ref, ref, config['merge_radius'])
            )

        # Collect the results for each channel as they become available
        for ch, fut in tqdm(zip(recording.channel_ids, futures), 'Spike Detection',
                            len(futures), unit='channel'):
            results[ch] = fut.result()

    return results
