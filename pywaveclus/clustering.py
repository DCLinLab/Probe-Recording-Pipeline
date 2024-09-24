# clustering.py
# spclustering: Super Paramagnetic Clustering Wrapper
from spclustering import SPC, plot_temperature_plot
import yaml


def load_clustering_config(config_file):
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        return config['clustering']


def SPC_clustering(features, config_file):
    config = load_clustering_config(config_file)
    min_clus = config['min_clus']
    plot_temperature = config['plot_temperature']

    clustering = SPC(mintemp=config['min_temp'], maxtemp=config['max_temp'], tempstep=config['temp_step'],
                     swcycles=config['sw_cycles'], nearest_neighbours=config['knn'], randomseed=config['rand_seed'])

    labels = {}
    metadata = {}

    for channel_id, feature in features.items():
        labels[channel_id], metadata[channel_id] = clustering.fit(feature, min_clus, return_metadata=True)

        if plot_temperature:
            plot_temperature_plot(metadata[channel_id])
            

    return labels, metadata
