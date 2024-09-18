import spikeinterface.preprocessing as spre
from .rm_mechanical_noise import rm_mechanical_noise


def clean_channels_by_imp(recording, imp, imp_thr=5e6):
    return recording.remove_channels([i for i in recording.get_channel_ids() if imp[i] > imp_thr])


def filter_guosong(recording, pass_band=(250, 6000), mov_window=2000, interval=100):
    recording = spre.bandpass_filter(recording, pass_band[0], pass_band[1],
                                     filter_order=4, filter_mode='sos', ftype='butter')
    recording = rm_mechanical_noise(recording, mov_window=mov_window, interval=interval)
    return recording


def filter_detailed(recording):
    """
    common_reference can
    suppress collective interference including mechanical noise.

    :param recording:
    :param pass_band:
    :return:
    """
    r_sort = spre.bandpass_filter(recording, 300, 6000,         # default in spike interface
                              filter_order=4, filter_mode='sos', ftype='butter') # less distortion in time
    r_detect = spre.bandpass_filter(recording, 300, 3000,       # used by waveclus3
                              filter_order=4, filter_mode='sos', ftype='ellip', rp=0.1, rs=40)  # better denoising

    # common ref removes collective activities including mechanical noise
    r_sort = spre.common_reference(recording=r_sort, operator="median", reference="global")
    r_detect = spre.common_reference(recording=r_detect, operator="median", reference="global")

    return r_sort, r_detect
