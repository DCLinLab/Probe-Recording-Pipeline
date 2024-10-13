from main import SortingCommands

from pathlib import Path
from traceback import print_exc


if __name__ == '__main__':
    pkl_dir = Path(r'G:\wavclus_result\curved_120')
    sessions = sorted(pkl_dir.rglob('*.pkl.tgz'))
    for s in sessions:
        try:
            SortingCommands().plot_waveforms(s, s.parent / 'waveforms')
            # SortingCommands().plot_quality_metrics(s, s.parent / 'quality_metrics.png')
            # SortingCommands().plot_auto_correlagrams(s, s.parent / 'auto_correlagrams.png')
        except:
            print_exc()
