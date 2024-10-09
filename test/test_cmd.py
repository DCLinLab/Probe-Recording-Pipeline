import unittest
from main import SortingCommands
from pathlib import Path
from extra.io import load_results
import numpy as np
from extra.plot import *
tst_folder = Path(r'D:\Yongzhi_Sun\03_Processed_Data\waveclus_240103\curved\20240330\w15\data')
# tst_folder = Path(r'C:\Users\LinLab_Workstation3\Desktop\test')


class MyTestCase(unittest.TestCase):

    def test_sorting(self):
        imp = r'D:\Yongzhi_Sun\03_Processed_Data\waveclus_240103\curved\20240330\imp\imp-w15.csv'
        SortingCommands().sorting(
            tst_folder, imp, tst_folder / 'out.pkl.xz', time_range=(3, 63)
        )

    def test_plot(self):
        a = load_results(tst_folder / r'out.pkl.xz')
        clusters = {}
        res = a['A-001']
        for i, clust in enumerate(res['labels']):
            if clust not in clusters:
                clusters[clust] = []
            clusters[clust].append(i)
        tot = len(res['waveforms'][0])
        for i, c in clusters.items():
            data = {
                'amp': np.concatenate([res['waveforms'][j] for j in c]),
                'time': np.tile(np.linspace(0, tot - 1, tot), len(c)) / 20000 * 1000,
            }
            plot_waveform(data)
        plt.show()

if __name__ == '__main__':
    unittest.main()
