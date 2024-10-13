from pathlib import Path
import shutil
from extra import io
import spikeinterface as si
from extra.preprocess import filter_detailed, clean_channels_by_imp
import scipy


if __name__ == '__main__':
    pkl_dir = Path(r'G:\wavclus_result\curved_120')
    new = Path(r'D:\Yongzhi_Sun\01_Raw_Data\Yongzhi_Sun\intan\curved_120')
    for i in pkl_dir.rglob('*.tgz'):
        # new_name =  str(i) + '.bak'
        # shutil.move(i, new_name)
        data = io.load_results(i,40)
        print(f'{i} loaded.')
        # recording = io.rhd_load(new / i.parent.relative_to(pkl_dir))
        # print('preprocess...')
        # time_range = [3, 63]
        # if time_range[0] < 0:
        #     time_range[0] = 0
        # tot_frame = recording.get_num_frames()
        # t0 = round(time_range[0] * 60 * recording.sampling_frequency)
        # t1 = round(time_range[1] * 60 * recording.sampling_frequency)
        # if t0 < 0:
        #     t0 = max(0, tot_frame + t0 + 1)
        # t0 = min(t0, tot_frame)
        # if t1 < 0:
        #     t1 = max(0, tot_frame + t1 + 60 * recording.sampling_frequency)
        # t1 = min(t1, tot_frame)
        # recording = recording.frame_slice(start_frame=t0, end_frame=t1)
        # r1, r2 = filter_detailed(recording)
        # r1 = r1.save(format='memory', n_jobs=8, progress_bar=True, chunk_duration='120s')
        # m = data['metadata']
        # del data['metadata']
        for c, v in data.items():
            # v['noise'] = si.get_noise_levels(r1.select_channels([c]), return_scaled=True)
            v['noise'] = v['noise'][0]
            # m = v['metadata']
            # for k in m:
            #     v[k] = m[k]
            # del v['metadata']
        io.save_results(data, i, 40)