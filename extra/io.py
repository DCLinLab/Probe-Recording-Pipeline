import pandas as pd
import pickle
import lzma
from pathlib import Path
import spikeinterface.extractors as se
from spikeinterface import concatenate_recordings


def load_intan_impedance(path, column='Impedance Magnitude at 1000 Hz (ohms)', to_omit=()):
    """
    sort the impedance from 0, removing some of them and let the number continue

    :param path:
    :return: a dict mapping channel id to impedance value
    """
    imped_tab = pd.read_csv(path).sort_values(by='Channel Number', ascending=True)
    # j = 0
    # out = {}
    # for ind, row in imped_tab.iterrows():
    #     if row['Channel Number'] in to_omit:
    #         continue
    #     out[str(j)] = row[column]
    #     j += 1
    return dict(zip(imped_tab['Channel Number'], imped_tab[column]))


def save_results(results, path):
    with lzma.open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path):
    with lzma.open(path, "rb") as compressed_file:
        # Use pickle.load to deserialize the data
        return pickle.load(compressed_file)


def rhd_load(folder, stream_name='RHD2000 amplifier channel'):
    files = sorted(Path(folder).glob('*.rhd'))
    files = [se.read_intan(i, stream_name=stream_name) for i in files]
    return concatenate_recordings(files)
