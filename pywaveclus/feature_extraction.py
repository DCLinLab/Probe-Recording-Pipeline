# feature_extraction.py
import numpy as np
import pywt
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import lilliefors
from tqdm import tqdm
import yaml


def load_feature_extraction_config(config_file):
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        return config['feature_extraction']


def pca_feature_extraction_for_channel(waveforms):
    return PCA().fit_transform(waveforms)[:, 1:11 ]  # Fit PCA and transform the data


def wavelet_feature_extraction_for_channel(waveforms, scales, min_inputs, max_inputs):
    nspk, ls = waveforms.shape
    cc = np.zeros((nspk, ls))

    try:
        spikes_l = waveforms.T.flatten()
        c_l = pywt.wavedec(spikes_l, 'haar', level=scales)
        l_wc = [len(c) for c in c_l]
        c_l = np.concatenate(c_l)
        wv_c = np.concatenate(([0], np.cumsum(l_wc[:-1])))
        nc = wv_c // nspk
        wccum, nccum = np.cumsum(wv_c), np.cumsum(nc)

        for cf in range(1, scales + 1):
            cc[:, int(nccum[cf - 1]):int(nccum[cf])] = np.reshape(
                c_l[int(wccum[cf - 1]):int(wccum[cf])], (nc[cf], nspk)).T
    except:
        for i in range(nspk):
            c_l = pywt.wavedec(waveforms[i, :], 'haar', level=scales)
            cc[i] = np.reshape(c_l, -1)[:ls]

    ks = []
    for i in range(ls):
        thr_dist = np.std(cc[:, i]) * 3
        thr_dist_min = np.mean(cc[:, i]) - thr_dist
        thr_dist_max = np.mean(cc[:, i]) + thr_dist
        aux = cc[(cc[:, i] > thr_dist_min) & (cc[:, i] < thr_dist_max), i]
        ks.append(lilliefors(aux, dist='norm', pvalmethod='table')[0] if len(aux) > 10 else 0)

    if isinstance(max_inputs, float):
        max_inputs = int(max_inputs * ls)
    ind = np.argsort(ks)[-max_inputs:]
    A = np.array(ks)[ind]
    nd = 10
    d = (A[nd - 1:] - A[:-nd + 1]) * max_inputs / (A.max() * nd)
    all_above1 = np.where(d >= 1)[0]

    inputs = min_inputs
    if len(all_above1) > 2:
        aux2 = np.diff(all_above1)
        temp_bla = np.convolve(aux2, np.ones(3) / 3, mode='same')
        temp_bla[0] = aux2[0]
        temp_bla[-1] = aux2[-1]
        find = np.where(temp_bla[1:] == 1)[0]
        if len(find) > 0:
            thr_knee_diff = all_above1[find[0]] + nd // 2
            inputs = max_inputs - thr_knee_diff + 1

    coeff = ind[-inputs:]
    inspk = np.zeros((nspk, inputs))
    for i in range(nspk):
        for j in range(inputs):
            inspk[i, j] = cc[i, coeff[j]]
    return inspk


def feature_extraction(waveforms, config_file):
    config = load_feature_extraction_config(config_file)
    features = {}
    if config['method'] == 'wav':       # wavelet tends to give to detailed results
        for channel, spike_waveforms in tqdm(waveforms.items()):
            features[channel] = wavelet_feature_extraction_for_channel(
                spike_waveforms, config['scales'], config['min_inputs'], config['max_inputs'])
    else:
        for channel, spike_waveforms in tqdm(waveforms.items()):
            features[channel] = pca_feature_extraction_for_channel(spike_waveforms)
    return features

