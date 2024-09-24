import unittest
from main import SortingCommands
from pathlib import Path
tst_folder = Path(r'C:\Users\LinLab_Workstation3\Desktop\test')


class MyTestCase(unittest.TestCase):

    def test_something(self):
        imp = list(tst_folder.glob('*.csv'))[0]
        SortingCommands().sorting(
            tst_folder, imp, tst_folder / 'out.pkl.xz', max_workers=12
        )

if __name__ == '__main__':
    unittest.main()
