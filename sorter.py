import fire
import numpy as np

from extra.preprocess import filter_detailed, clean_channels_by_imp
from pywaveclus.spike_detection import detect_spikes
import shutil
from pywaveclus.clustering import clustering
from extra.io import rhd_load, save_results, load_intan_impedance, load_results
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from extra.plot import plot_quality_metrics, plot_waveform, plot_auto_correlagrams
from pywaveclus.feature_extraction import feature_extraction


def waveclus_pipeline(recording, config, imps=None, cache='', max_workers=None):
    nj = 4
    if max_workers is not None:
        assert max_workers > 0
        nj = min(max_workers, nj)
    type = 'memory'
    if cache:
        type = 'zarr'
        if Path(cache).exists():
            shutil.rmtree(cache, ignore_errors=True)
    r_sort, r_detect = filter_detailed(recording)
    r_sort = r_sort.save(format=type, folder=f'{cache}/r1', n_jobs=nj, progress_bar=True, chunk_duration='120s')
    r_detect = r_detect.save(format=type, folder=f'{cache}/r2', n_jobs=nj, progress_bar=True, chunk_duration='120s')
    results = detect_spikes(recording, r_sort, r_detect, max_workers, **config['spike_detection'])  # channel by channel spike timestamps & indices
    features = feature_extraction(results, **config['feature_extraction'])
    labels = clustering(features, max_workers, **config['clustering'])  # channel by channel labels
    for ch, res in results.items():
        res['labels'] = labels[ch]
        if imps is not None:
            res['imp'] = imps[ch]
    return results


class WaveclusInterface:
    def __init__(self, max_workers=0, config=Path(__file__).parent / 'config.yaml', cache=''):
        self.max_workers = None
        self._cache = cache
        if max_workers != 0:
            self.max_workers = max_workers
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

    def sort(self, rhd_dir, save_to='', impedance_file='', time_range=(0, -1)):
        """
            A standard pipeline based on waveclus3, difference is:
            1. for spike detection use elliptic filtering, for spike alignment use butterworth filtering
            2. use common reference for removal of unwanted interference

            :param rhd_dir:
            :param save_to:
            :param impedance_file:
            :param time_range: time range of the recording, in minutes.
            :param cache:
            :return: results as dict indexed by channel, each item containing spikes and their waveforms, cluster labels, etc.
            """
        recording = rhd_load(rhd_dir)
        imps = None
        if impedance_file:
            thr = self.config['preprocessing']['impedance_thr']
            print(f'Load impedance and filter channels over {thr} Ohm')
            imps = load_intan_impedance(impedance_file)
            recording = clean_channels_by_imp(recording, imps, thr)
        tot_frame = recording.get_num_frames()
        t0 = round(time_range[0] * 60 * recording.sampling_frequency)
        t1 = round(time_range[1] * 60 * recording.sampling_frequency)
        if t0 < 0:
            t0 = max(0, tot_frame + t0 + 1)
        t0 = min(t0, tot_frame)
        if t1 < 0:
            t1 = max(0, tot_frame + t1 + 60 * recording.sampling_frequency)
        t1 = min(t1, tot_frame)
        print(f'Selected frame range: [{t0}, {t1}]')
        recording = recording.frame_slice(start_frame=t0, end_frame=t1)
        results = waveclus_pipeline(recording, self.config, imps, self._cache, self.max_workers)
        if save_to:
            save_results(results, save_to)
        return results

    @staticmethod
    def _sort_results(res):
        clusters = {}
        for i, clust in enumerate(res['labels']):
            if clust == 0:
                continue
            if clust not in clusters:
                clusters[clust] = []
            clusters[clust].append(i)
        return clusters

    def plot_waveforms(self, results, outdir, suffix='.png'):
        """

        :param results:
        :param outdir:
        :param suffix:
        :return:
        """
        if not isinstance(results, dict):
            results = load_results(results, None)
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        for ch, res in results.items():
            clusters = self._sort_results(res)
            ls = len(res['waveforms'][0])
            for i, c in clusters.items():
                data = {
                    'amp': np.concatenate([res['waveforms'][j] for j in c]),
                    'time': np.tile(np.linspace(0, ls - 1, ls), len(c)) / res['fs'] * 1000,
                }
                plot_waveform(data, i, int(self.config['spike_detection']['w_pre']), len(c) / res['dur'])
            plt.savefig(outdir / f'{ch}{suffix}', bbox_inches='tight')
            plt.close()

    def plot_quality_metrics(self, results, outpath, snr_thr=4, fr_thr=.1):
        """

        :param snr_thr:
        :param fr_thr:
        :return:
        """
        if not isinstance(results, dict):
            results = load_results(results, None)
        snr = []
        fr = []
        imp = []
        peak = []
        for ch, res in results.items():
            clusters = self._sort_results(res)
            for i, c in clusters.items():
                r = len(c) / res['dur']
                waveforms = np.array([res['waveforms'][j] for j in c])
                signal = np.abs(waveforms.mean(axis=0))[self.config['spike_detection']['w_pre']]
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
        plt.savefig(outpath , bbox_inches='tight')
        plt.close()

    def plot_auto_correlagrams(self, results, outdir, bin_time=1, max_lag=50, suffix='.png'):
        """

        :param outdir:
        :param results:
        :param bin_time:
        :param max_lag:
        :param suffix:
        :return:
        """
        if not isinstance(results, dict):
            results = load_results(results, None)
        outdir = Path(outdir)
        Path(outdir).mkdir(exist_ok=True, parents=True)
        for ch, res in results.items():
            clusters = self._sort_results(res)
            for i, c in clusters.items():
                data = [res['times'][j] for j in c]
                plot_auto_correlagrams(data, res['fs'], bin_time, max_lag)
                plt.savefig(outdir / f'{ch}_{i}{suffix}', bbox_inches='tight')
                plt.close()

    def plot_phase_locking(self, rhd_dir, results, outdir, suffix='.png'):
        pass


if __name__ == '__main__':
    fire.Fire(WaveclusInterface)