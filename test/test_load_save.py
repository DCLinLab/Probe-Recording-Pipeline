import unittest
from extra.io import *

tst_folder = Path(r'C:\Users\LinLab_Workstation3\Desktop\test')


class MyTestCase(unittest.TestCase):
    def test_intan_load(self):
        rhd = rhd_load(tst_folder)
        print(rhd)

    def test_imp_load(self):
        p = tst_folder / 'imp-w15.csv'
        imp = load_intan_impedance(p, to_omit=['A-048', 'A-049', 'A-050'])
        print(imp)


if __name__ == '__main__':
    unittest.main()
