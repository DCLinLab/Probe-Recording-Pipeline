import pandas as pd
import pickle
import gzip
import lzma
import tarfile
from pathlib import Path
import spikeinterface.extractors as se
from spikeinterface import concatenate_recordings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import io
from functools import partial


def load_intan_impedance(path, column='Impedance Magnitude at 1000 Hz (ohms)'):
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


def compress_chunk(data):
    serialized_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with gzip.GzipFile(fileobj=buf, mode='wb') as f:
            f.write(serialized_data)
        content = buf.getvalue()
    return content


def decompress_chunk(gz):
    with gzip.GzipFile(fileobj=io.BytesIO(gz), mode='rb') as f:
        return pickle.loads(f.read())


def save_results(results, path, max_workers=None):
    channels = list(results.keys())
    values = list(results.values())
    gz_chunks = []
    loader = partial(tqdm, desc='GZIP compressing results', unit='channel')
    if max_workers == 1:
        for v in loader(values):
            gz_chunks.append(compress_chunk(v))
    else:
        with ProcessPoolExecutor(max_workers) as executor:
            futures = []
            for i in values:
                futures.append(executor.submit(compress_chunk, i))
            for i in loader(futures):
                gz_chunks.append(i.result())
    with tarfile.open(path, 'w:gz') as t:
        for ch, gz in zip(channels, gz_chunks):
            tar_info = tarfile.TarInfo(name=f'{ch}.pkl.gz')
            tar_info.size = len(gz)
            t.addfile(tar_info, io.BytesIO(gz))


def load_results(path, max_workers=None):
    with tarfile.open(path, 'r:gz') as t:
        members = t.getmembers()
        loader = partial(tqdm, desc='GZIP decompressing results', unit='channel')
        if max_workers == 1:
            data = {}
            for m in loader(members):
                gz = t.extractfile(m).read()
                channel = m.name.removesuffix('.pkl.gz')
                data[channel] = decompress_chunk(gz)
        else:
            to_decompress = []
            channels = []
            for m in members:
                to_decompress.append(t.extractfile(m).read())
                channels.append(m.name.removesuffix('.pkl.gz'))
            with ProcessPoolExecutor(max_workers) as executor:
                futures, results = [], []
                for i in to_decompress:
                    futures.append(executor.submit(decompress_chunk, i))
                for fut in loader(futures):
                    results.append(fut.result())
                data = dict(zip(channels, results))
    return data


def save_results_old(results, path):
    with lzma.open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results_old(path):
    with lzma.open(path, "rb") as compressed_file:
        # Use pickle.load to deserialize the data
        return pickle.load(compressed_file)


def rhd_load(folder, stream_name='RHD2000 amplifier channel'):
    files = sorted(Path(folder).glob('*.rhd'))
    assert len(files) > 0, "No intan RHD2000 recording found in this directory."
    files = [se.read_intan(i, stream_name=stream_name, use_names_as_ids=True)
             for i in tqdm(files, 'Loading RHD2000 files')]
    return concatenate_recordings(files)

