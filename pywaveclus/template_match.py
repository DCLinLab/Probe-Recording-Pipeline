import yaml
import numpy as np


def build_templates(classes, features):
    templates = []
    maxdist = []
    for i in np.unique(classes):
        f = features[classes == i]
        templates.append(f.mean(axis=0))
        maxdist.append(f.var(axis=0).sum() ** .5)
    return templates, maxdist


def nearest_neighbor(x, vectors, maxdist):
    """

    :param x: features of a spike
    :param vectors: features of templates
    :param maxdist: thr for each template
    :return:
    """
    # Calculate distances from x to all templates
    dists = np.linalg.norm(x - vectors, axis=1)

    # Find conforming templates within maxdist
    conforming = np.where(dists < maxdist)[0]

    # If no conforming neighbors, return 0
    index = 0
    if len(conforming) != 0:
        i = dists[conforming].argmin()
        index = conforming[i]
    return index



def force_membership(temp_spikes, temp_class, spikes_to_match, config):
    type = config['type']
    nspk = len(spikes_to_match)
    res = []
    if type == 'nn':
        raise NotImplementedError
    elif type == 'center':
        centers, sd = build_templates(temp_class, temp_spikes)
        sdnum = config['sdnum']
        for i in range(nspk):
            res.append(nearest_neighbor(spikes_to_match[i], centers, sdnum * sd))
    elif type == 'ml':
        raise NotImplementedError
    elif type == 'mahal':
        raise NotImplementedError
    else:
        raise LookupError(f"template matching method {type} not found.")
    return res