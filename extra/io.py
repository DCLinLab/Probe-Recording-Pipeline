import pandas as pd
import pickle
import lzma
from glob import glob
import spikeinterface.extractors as se
from spikeinterface import concatenate_recordings


def load_intan_impedance(path, column='Impedance Magnitude at 1000 Hz (ohms)'):
    """

    :param path:
    :return: a dict mapping channel id to impedance value
    """
    imped_tab = pd.read_csv(path).sort_values(by='Channel Number', ascending=True)
    imped_tab['ind'] = [str(i) for i in imped_tab.index]
    imped_tab.set_index('ind', inplace=True)
    return imped_tab[column].to_dict()


def save_results(results, path):
    with lzma.open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path):
    with lzma.open(path, "rb") as compressed_file:
        # Use pickle.load to deserialize the data
        return pickle.load(compressed_file)


def rhd_load(prefix, stream_name='RHD2000 amplifier channel'):
    files = sorted(glob(prefix + '*.rhd'))
    files = [se.read_intan(i, stream_name=stream_name) for i in files]
    return concatenate_recordings(files)
