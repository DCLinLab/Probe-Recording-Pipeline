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

    clustering = SPC(mintemp=0, maxtemp=0.251)

    labels = {}
    metadata = {}

    for channel_id, feature in features.items():
        labels[channel_id], metadata[channel_id] = clustering.fit(feature, min_clus, return_metadata=True)

        if plot_temperature:
            plot_temperature_plot(metadata[channel_id])
            

    return labels, metadata
