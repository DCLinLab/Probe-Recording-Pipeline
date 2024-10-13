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

    def test_new_save(self):
        a = load_results_old(r'C:\Users\LinLab_Workstation3\Desktop\test\out.pkl.xz')
        save_results(a, r'C:\Users\LinLab_Workstation3\Desktop\test\out.tar.gz', 20)

    def test_new_load(self):
        a = load_results(r'C:\Users\LinLab_Workstation3\Desktop\test\out.tar.gz', None)
        print(a)


if __name__ == '__main__':
    unittest.main()
