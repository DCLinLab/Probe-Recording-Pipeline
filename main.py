import fire
import numpy as np

import spikeinterface as si
from extra.preprocess import filter_detailed, clean_channels_by_imp
from pywaveclus.spike_detection import detect_spikes
import shutil
from pywaveclus.waveform_extraction import extract_waveforms
from pywaveclus.clustering import clustering
from extra.io import rhd_load, save_results, load_intan_impedance, load_results
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from extra.plot import plot_quality_metrics, plot_waveform, plot_auto_correlagrams
from pywaveclus.feature_extraction import feature_extraction


class SortingCommands:
    def __init__(self, max_workers=0, config_file=Path(__file__).parent / 'config.yaml'):
        self.max_workers = None
        if max_workers != 0:
            self.max_workers = max_workers
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def sorting(self, rhd_folder: str, impedance_file: str, out_path='sorted.pkl.xz', time_range=(0, -1), cache=''):
        """
            A standard pipeline based on waveclus3, difference is:
            1. for spike detection use elliptic filtering, for spike alignment use butterworth filtering
            2. use common reference for removal of unwanted interference

            :param rhd_folder:
            :param impedance_file:
            :param recording: a raw recording
            :param out_path: output path to the results file (lzma compressed pickle)
            :param time_range: time range of the recording, in minutes.
            :param config_file: yaml config file
            :return: results as dict indexed by channel, each item containing spikes and their waveforms, cluster labels, etc.
            """

        recording = rhd_load(rhd_folder)
        print('preprocess...')
        thr = self.config['preprocessing']['impedance_thr']
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
            t1 = max(0, tot_frame + t1 + 60 * recording.sampling_frequency)
        t1 = min(t1, tot_frame)
        recording = recording.frame_slice(start_frame=t0, end_frame=t1)
        r_sort, r_detect = filter_detailed(recording)
        if self.max_workers is None:
            nj = 8
        else:
            nj = min(self.max_workers, 8)
        if cache == '':
            type = 'memory'
        else:
            type = 'zarr'
            if Path(cache).exists():
                shutil.rmtree(cache)
        r_sort = r_sort.save(format=type,  folder=f'{cache}/r1', n_jobs=nj, progress_bar=True, chunk_duration='120s')
        r_detect = r_detect.save(format=type,  folder=f'{cache}/r2', n_jobs=nj, progress_bar=True, chunk_duration='120s')
        results = detect_spikes(recording, r_sort, r_detect, self.max_workers, **self.config['spike_detection'])  # channel by channel spike timestamps & indices
        waveforms = extract_waveforms(results, r_sort, nj, **self.config['extract_waveform'])  # channel by channel #spike * 64 waveforms
        features = feature_extraction(waveforms, **self.config['feature_extraction'])
        labels = clustering(features, None, **self.config['clustering'])  # channel by channel labels
        for ch, res in results.items():
            res['labels'] = labels[ch]
            res['waveforms'] = waveforms[ch]
            res['imp'] = imps[ch]
            res['fs'] = recording.sampling_frequency
            res['dur'] = recording.get_duration()
            res['noise'] = si.get_noise_levels(r_sort.select_channels([ch]), return_scaled=True)[0]
        save_results(results, out_path)

    def plot_waveforms(self, sorting_result: str, out_dir='.', suffix='.png'):
        """

        :param sorting_result:
        :param out_dir:
        :param suffix:
        :return:
        """
        results = load_results(sorting_result, None)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
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
                    'time': np.tile(np.linspace(0, tot - 1, tot), len(c)) / res['fs'] * 1000,
                }
                plot_waveform(data, i, int(self.config['extract_waveform']['w_pre']) - 1, len(c) / res['dur'])
            plt.savefig(out_dir / f'{ch}{suffix}', bbox_inches='tight')
            plt.close()

    def plot_quality_metrics(self, sorting_result: str, out_path: str, snr_thr=4, fr_thr=.1):
        """

        :param sorting_result:
        :param out_path:
        :return:
        """
        results = load_results(sorting_result, None)
        snr = []
        fr = []
        imp = []
        peak = []
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust == 0:
                    continue
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            for i, c in clusters.items():
                r = len(c) / res['dur']
                waveforms = np.array([res['waveforms'][j] for j in c])
                signal = np.abs(waveforms.mean(axis=0))[self.config['extract_waveform']['w_pre']]
                s = signal / res['noise']
                if r < fr_thr or s < snr_thr:
                    continue
                fr.append(r)
                snr.append(s)
                imp.append(res['imp'] / 10000)
                peak.append(signal)
        if len(fr) == 0:
            print('No cluster passed quality control.')
            return
        plot_quality_metrics(snr, peak, fr, imp, len(results))
        plt.savefig(out_path , bbox_inches='tight')
        plt.close()

    def plot_auto_correlagrams(self, sorting_result: str, out_dir='.', bin_time=1, max_lag=50, suffix='.png'):
        """

        :param sorting_result:
        :param bin_time:
        :param max_lag:
        :param out_dir:
        :param suffix:
        :return:
        """
        results = load_results(sorting_result, None)
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        for ch, res in results.items():
            clusters = {}
            for i, clust in enumerate(res['labels']):
                if clust == 0:
                    continue
                if clust not in clusters:
                    clusters[clust] = []
                clusters[clust].append(i)
            for i, c in clusters.items():
                data = [res['spikes_time'][j] for j in c]
                plot_auto_correlagrams(data, res['fs'], bin_time, max_lag)
                plt.savefig(Path(out_dir) / f'{ch}_{i}{suffix}', bbox_inches='tight')
                plt.close()

    def plot_phase_locking(self, sorting_result: str, rhd_prefix: str, out_dir='.', suffix='.png'):
        pass


if __name__ == '__main__':
    fire.Fire(SortingCommands)