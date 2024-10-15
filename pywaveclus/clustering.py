# clustering.py
# spclustering: Super Paramagnetic Clustering Wrapper
from spclustering import SPC
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
from pywaveclus.template_match import force_membership


def clustering_for_channel(features, config):
    spc = SPC(config['min_temp'], config['max_temp'], config['temp_step'],
              config['sw_cycles'], config['knn'], randomseed=config['rand_seed'])
    random.seed(config['rand_seed'])
    if len(features) > config['max_spk']:
        ipermut = random.sample(range(len(features)), config['max_spk'])
        irest = list(set(range(len(features))) - set(ipermut))
    else:
        ipermut = random.sample(range(len(features)), len(features))
        irest = []
    inspk = features[ipermut]
    labels_temp = spc.fit(inspk, min_clus=config['min_clus'], elbow_min=config['elbow_min'], c_ov=config['c_ov'])
    labels = [0] * len(features)
    for i, j in zip(ipermut, labels_temp):
        labels[i] = j
    if irest:
        to_match = features[irest]
        more_labels = force_membership(inspk, labels_temp, to_match, config['template_match'])
        for i, j in zip(irest, more_labels):
            labels[i] = j
    return labels


def clustering(features, max_workers, **config):
    labels = {}
    with ProcessPoolExecutor(max_workers) as executor:
        futures, results = [], []
        for ch, f in features.items():
            if len(f) == 0:
                labels[ch] = []
            else:
                futures.append(executor.submit(clustering_for_channel, f, config))
        for fut in tqdm(futures, 'Clustering', unit='channel'):
            results.append(fut.result())
    return dict(zip(features.keys(), results))
