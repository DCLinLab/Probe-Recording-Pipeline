import fire
import numpy as np

from extra.preprocess import filter_detailed, clean_channels_by_imp
from pywaveclus.spike_detection import detect_spikes
from pywaveclus.feature_extraction import feature_extraction
from pywaveclus.waveform_extraction import extract_waveforms
from pywaveclus.clustering import SPC_clustering
from extra.io import rhd_load, save_results, load_intan_impedance, load_results
from pathlib import Path
import yaml
import seaborn as sns
import matplotlib.pyplot as plt


class SortingCommands:

    def sorting(self, rhd_prefix: str, impedance_file: str, out_path='sorted.pkl.xz', time_range=(0, -1),
                config_file=Path(__file__).parent / 'config.yaml',
               max_workers=0):
        """
            A standard pipeline based on waveclus3, difference is:
            1. for spike detection use elliptic filtering, for spike alignment use butterworth filtering
            2. use common reference for removal of unwanted interference

            :param recording: a raw recording
            :param out_path: output path to the results file (lzma compressed pickle)
            :param time_range: time range of the recording, in seconds.
            :param config_file: yaml config file
            :return: results as dict indexed by channel, each item containing spikes and their waveforms, cluster labels, etc.
            """

        recording = rhd_load(rhd_prefix)
        print('preprocess...')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            thr = config['preprocessing']['impedance_thr']
        recording = clean_channels_by_imp(recording, load_intan_impedance(impedance_file), thr)
        if time_range[0] < 0:
            time_range[0] = 0
        tot_frame = recording.get_num_frames()
        t0 = round(time_range[0] * 60 * recording.sampling_frequency)
        t1 = round(time_range[1] * 60 * recording.sampling_frequency)
        if t0 < 0:
            t0 = max(0, tot_frame + t0 + 1)
        t0 = min(t0, tot_frame)
        if t1 < 0:
            t1 = max(0, tot_frame + t1 + 1)
        t1 = min(t1, tot_frame)
        recording = recording.frame_slice(start_frame=t0, end_frame=t1)
        r1, r2 = filter_detailed(recording)
        print('detect spikes...')
        results = detect_spikes(recording, r1, r2, config_file, max_workers)  # channel by channel spike timestamps & indices
        print('extract waveforms...')
        waveforms = extract_waveforms(results, r1, config_file)  # channel by channel #spike * 64 waveforms
        print('extract features')
        features = feature_extraction(waveforms, config_file)
        print('do clustering')
        labels = SPC_clustering(features, config_file)[0]  # channel by channel labels
        for ch, res in results.items():
            res['labels'] = labels[ch]
            res['waveforms'] = waveforms[ch]
        results['metadata'] = {}
        results['metadata']['fs'] = recording.sampling_frequency
        save_results(results, out_path)

    def plot_waveforms(self, sorting_result: str, out_dir='.', format='.png'):
        """

        :param sorting_result:
        :return:
        """
        results = load_results(sorting_result)
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            tot = len(res['waveforms'][0])
            for i, c in clusters.items():
                data = {
                    'amp': np.concatenate([res['waveforms'][i] for i in c]),
                    'time': np.tile(np.linspace(0, tot - 1, tot), len(c)) / results['metadata']['fs'] * 1000,
                }
                sns.lineplot(x="time", y="amp", data=data)
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')
                plt.savefig(Path(out_dir) / f'ch_{ch}_clust_{i}{format}', bbox_inches='tight')

    def plot_quality_metrics(self, sorting_result: str, out_dir='.', format='.png'):
        results = load_results(sorting_result)
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            tot = len(res['waveforms'][0])
            for i, c in clusters.items():
                data = {
                    'amp': np.concatenate([res['waveforms'][i] for i in c]),
                    'time': np.tile(np.linspace(0, tot - 1, tot), len(c)) / results['metadata']['fs'] * 1000,
                }
                sns.lineplot(x="time", y="amp", data=data)
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')
                plt.savefig(Path(out_dir) / f'ch_{ch}_clust_{i}{format}', bbox_inches='tight')


if __name__ == '__main__':
    fire.Fire(SortingCommands)