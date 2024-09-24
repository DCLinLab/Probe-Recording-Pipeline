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
import matplotlib.pyplot as plt
from extra.plot import plot_quality_metrics, plot_waveform, plot_auto_correlagrams


class SortingCommands:

    def sorting(self, rhd_prefix: str, impedance_file: str, out_path='sorted.pkl.xz', time_range=(0, -1),
                config_file=Path(__file__).parent / 'config.yaml',
               max_workers=0):
        """
            A standard pipeline based on waveclus3, difference is:
            1. for spike detection use elliptic filtering, for spike alignment use butterworth filtering
            2. use common reference for removal of unwanted interference

            :param rhd_prefix:
            :param impedance_file:
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
        imps = load_intan_impedance(impedance_file)
        recording = clean_channels_by_imp(recording, imps, thr)
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
            res['imp'] = imps[ch]
        results['metadata'] = {}
        results['metadata']['fs'] = recording.sampling_frequency
        results['metadata']['dur'] = recording.get_duration()
        save_results(results, out_path)

    def plot_waveforms(self, sorting_result: str, out_dir='.', suffix='.png'):
        """

        :param sorting_result:
        :param out_dir:
        :param suffix:
        :return:
        """
        results = load_results(sorting_result)
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust == 0:
                    continue
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            tot = len(res['waveforms'][0])
            for i, c in clusters.items():
                data = {
                    'amp': np.concatenate([res['waveforms'][j] for j in c]),
                    'time': np.tile(np.linspace(0, tot - 1, tot), len(c)) / results['metadata']['fs'] * 1000,
                }
                plot_waveform(data)
                plt.savefig(Path(out_dir) / f'ch_{ch}_clust_{i}{suffix}', bbox_inches='tight')

    def plot_quality_metrics(self, sorting_result: str, out_path: str):
        """

        :param sorting_result:
        :param out_path:
        :return:
        """
        results = load_results(sorting_result)
        snr = []
        fr = []
        imp = []
        peak = []
        nclust = []
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust == 0:
                    continue
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            nclust.append(len(clusters))
            for i, c in clusters.items():
                waveforms = np.array([res['waveforms'][j] for j in c])
                signal = waveforms.mean(axis=0).max()
                noise = waveforms.std()
                snr.append(signal / noise)
                fr.append(len(c) / res['metadata']['dur'])
                imp.append(res['imp'] / 10000)
                peak.append(signal)
        plot_quality_metrics(snr, fr, peak, imp, nclust)
        plt.savefig(out_path, bbox_inches='tight')

    def plot_auto_correlagrams(self, sorting_result: str, bin_time=1, max_lag=50, out_dir='.', suffix='.png'):
        """

        :param sorting_result:
        :param bin_time:
        :param max_lag:
        :param out_dir:
        :param suffix:
        :return:
        """
        results = load_results(sorting_result)
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust == 0:
                    continue
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            for i, c in clusters.items():
                data = res['spikes'][i]
                plot_auto_correlagrams(data, res['metadata']['fs'], bin_time, max_lag)
                plt.savefig(Path(out_dir) / f'ch_{ch}_clust_{i}{suffix}', bbox_inches='tight')

    def plot_phase_locking(self, sorting_result: str, rhd_prefix: str, out_dir='.', suffix='.png'):
        pass


if __name__ == '__main__':
    fire.Fire(SortingCommands)