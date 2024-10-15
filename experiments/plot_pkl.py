from sorter import WaveclusInterface

from pathlib import Path
from traceback import print_exc


if __name__ == '__main__':
    pkl_dir = Path(r'G:\wavclus_result\curved_120\20240911_m2')
    sessions = sorted(pkl_dir.rglob('*.pkl.tgz'))
    for s in sessions:
        try:
            # WaveclusInterface().plot_waveforms(s, s.parent / 'waveforms')
            WaveclusInterface().plot_quality_metrics(s, s.parent / 'quality_metrics.png')
            # WaveclusInterface().plot_auto_correlagrams(s, s.parent / 'auto_correlagrams.png')
        except:
            print_exc()
