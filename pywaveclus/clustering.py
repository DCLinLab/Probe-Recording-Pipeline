# clustering.py
# spclustering: Super Paramagnetic Clustering Wrapper
from spclustering import SPC
import yaml
from multiprocessing import Pool
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def load_clustering_config(config_file):
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        return config['clustering']


def spc_clustering_for_channel(features, min_clus, min_temp, max_temp, temp_step, sw_cycles, knn, rand_seed):
    spc = SPC(min_temp, max_temp, temp_step, sw_cycles, knn, randomseed=rand_seed)
    labels = spc.fit(features, min_clus=min_clus)
    return labels


def clustering(features, config_file):
    config = load_clustering_config(config_file)
    min_clus = config['min_clus']

    labels = {}
    with ProcessPoolExecutor() as executor:
        futures = []
        for channel_id, feature in features.items():
            if len(feature) == 0:
                labels[channel_id] = []
            else:
                futures.append(executor.submit(spc_clustering_for_channel, feature, min_clus,
                           config['min_temp'], config['max_temp'], config['temp_step'],
                            config['sw_cycles'], config['knn'], config['rand_seed']
                           ))
        for ch, fut in tqdm(zip(features, futures), total=len(features)):
            labels[ch] = fut.result()

    return labels
