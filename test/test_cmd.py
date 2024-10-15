import unittest
from sorter import WaveclusInterface
from pathlib import Path
import shutil


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.tst_folder = Path(r'D:\Yongzhi_Sun\03_Processed_Data\waveclus_240103\curved\20240330\w15\data')
        # self.tst_folder = Path(r'C:\Users\LinLab_Workstation3\Desktop\test')
        self.tst_folder = Path(r'C:\Users\LinLab_Workstation3\Desktop\test2')
        # self.imp = r'D:\Yongzhi_Sun\03_Processed_Data\waveclus_240103\curved\20240330\imp\imp-w15.csv'
        self.imp = r'D:\Yongzhi_Sun\01_Raw_Data\Yongzhi_Sun\intan\curved_120\20240911_m2\w4\imp-w4.csv'
        self.cmd = WaveclusInterface()

    def test_sorting(self):
        self.cmd.sort(self.tst_folder, self.tst_folder / 'out.pkl.tgz', self.imp, time_range=(0, 10))

    def test_plot(self):
        # shutil.rmtree(self.tst_folder / 'waveforms', ignore_errors=True)
        # self.cmd.plot_waveforms(self.tst_folder / 'out.pkl.tgz', self.tst_folder / 'waveforms')
        # self.cmd.plot_quality_metrics(self.tst_folder / 'out.pkl.tgz', self.tst_folder / 'quality_metrics.png')
        shutil.rmtree(self.tst_folder / 'auto_correlagrams', ignore_errors=True)
        self.cmd.plot_auto_correlagrams(self.tst_folder / 'out.pkl.tgz', self.tst_folder / 'auto_correlagrams')

if __name__ == '__main__':
    unittest.main()
