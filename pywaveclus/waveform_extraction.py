import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from spikeinterface import ChannelSliceRecording
from scipy.interpolate import splrep, splev
from tqdm import tqdm


def load_waveform_extraction_config(config_file):
    """Load waveform extraction configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file. Default is 'config.yaml'.

    Returns:
        dict: A dictionary containing the waveform extraction configuration parameters.
    """
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        return config['extract_waveform']


def extract_waveforms(results, recording_bp2, config_file, max_workers):
    """Extracts waveforms for detected spikes in all channels.

    Args:
        results (dict): A dictionary containing the spike detection results for each channel.
                        It can also be a single-channel result in the form of a dictionary.
        recording_bp2: The recording object for the channels' bandpass 2 data.
        config_file: Path to the YAML configuration file. Default is 'config.yaml'.

    Returns:
        dict: A dictionary where keys are the channel ids, and values are the extracted waveforms for detected spikes in each channel.
    """

    config = load_waveform_extraction_config(config_file)
    detect = config.get('detect_method', 'neg')
    w_pre = config.get('w_pre', 20)
    w_post = config.get('w_post', 44)
    int_factor = config.get('int_factor', 5)

    spikes_waveforms = {}
    if max_workers <= 0:
        max_workers = None
    with ProcessPoolExecutor(max_workers) as executor:
        # Submit tasks for each channel to the ThreadPoolExecutor
        futures = []
        for channel_id, result in results.items():
            futures.append(executor.submit(extract_waveforms_for_channel, result['spikes_time'],
                                           result['spikes_index'],
                           ChannelSliceRecording(recording_bp2, [channel_id]).get_traces(return_scaled=True),
                           w_pre, w_post, int_factor, detect))

        # Collect the results for each channel as they become available
        for ch, fut in tqdm(zip(results, futures), total=len(results)):
            spikes_waveforms[ch] = fut.result()

    return spikes_waveforms


def extract_waveforms_for_channel(spikes_times, indices, xf, w_pre, w_post, interp_factor, detect):
    nspk = len(spikes_times)
    ls = w_pre + w_post
    spikes_waveforms = np.zeros((nspk, ls))

    for i in range(nspk):
        extra = 2
        spike = xf[np.arange(-w_pre - extra, w_post + extra) + indices[i]]

        s = np.arange(len(spike))
        interp_x = np.arange(0, len(spike), 1 / interp_factor)
        tck = splrep(s, spike)
        interp_y = splev(interp_x, tck)

        if detect == 'pos':
            iaux = interp_y[int((w_pre + extra - 1) * interp_factor):int((w_pre + extra + 1) * interp_factor)].argmax()
        elif detect == 'neg':
            iaux = interp_y[int((w_pre + extra - 1) * interp_factor):int((w_pre + extra + 1) * interp_factor)].argmin()
        else:
            iaux = np.abs(
                interp_y[int((w_pre + extra - 1) * interp_factor):int((w_pre + extra + 1) * interp_factor)]).argmax()

        iaux += (w_pre + extra - 1) * interp_factor - 1

        spikes_waveforms[i, :] = interp_y[int(iaux - (w_pre - 1) * interp_factor):int(
            iaux + w_post * interp_factor + 1):interp_factor]

    return spikes_waveforms
