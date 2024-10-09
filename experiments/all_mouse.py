from main import SortingCommands
from pathlib import Path
from traceback import print_exc


if __name__ == '__main__':
    raw_dir = Path(r'D:\Yongzhi_Sun\01_Raw_Data\Yongzhi_Sun\intan\curved_120')
    res_dir = Path(r'G:\wavclus_result\curved_120')
    sessions = sorted(raw_dir.rglob('imp*.csv'))
    for s in sessions:
        new_folder = res_dir / s.parent.relative_to(raw_dir)
        new_folder.mkdir(exist_ok=True, parents=True)
        out =  new_folder / 'spikes.pkl.xz'
        if out.exists():
            print(f'Skip {s.parent}')
            continue
        print(f'Do {s.parent}')
        try:
            SortingCommands().sorting(s.parent, s, out, time_range=(3, 63), max_workers=20,
                                      config_file=Path(__file__).parent / 'config.yaml', cache=r'G:\cache')
        except:
            print_exc()
