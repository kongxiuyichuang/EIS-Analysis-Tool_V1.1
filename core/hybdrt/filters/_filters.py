import numpy as np
from scipy import ndimage

from ._scifilters import empty_gaussian_filter1d


def nonuniform_gaussian_filter1d(a, sigma, axis=-1, empty=False,
                                 mode='reflect', cval=0.0, truncate=4, order=0,
                                 sigma_node_factor=1.5, min_sigma=0.25):
    """Apply a 1-D Gaussian filter with a spatially varying sigma.

    The filter is applied at a set of logarithmically spaced sigma nodes and
    the results are blended with linearly decreasing weights in log(sigma)
    space.

    :param ndarray a: input array
    :param ndarray sigma: array of standard deviations, same shape as ``a``
    :param int axis: axis along which to filter
    :param bool empty: if True, apply an "empty" Gaussian filter (central
        weight set to zero)
    :param str mode: boundary mode (passed to scipy.ndimage.gaussian_filter1d)
    :param float cval: constant value for ``mode='constant'``
    :param float truncate: kernel truncation in standard deviations
    :param int order: derivative order of the filter
    :param float sigma_node_factor: factor between consecutive sigma nodes
    :param float min_sigma: minimum effective sigma
    :return ndarray: filtered array
    """
    if np.max(sigma) > 0:
        sigma = np.maximum(sigma, 1e-8)
        # Get sigma nodes
        min_ls = max(np.min(np.log10(sigma)), np.log10(min_sigma))  # Don't go below min effective value
        max_ls = max(np.max(np.log10(sigma)), np.log10(min_sigma))
        num_nodes = int(np.ceil((max_ls - min_ls) / np.log10(sigma_node_factor))) + 1
        sigma_nodes = np.logspace(min_ls, max_ls, num_nodes)

        if np.min(sigma) < min_sigma:
            # If smallest sigma is below min effective value, insert dummy node at lowest value
            # This node will simply return the original array

            # Determine factor for uniform node spacing
            if len(sigma_nodes) > 1:
                factor = sigma_nodes[-1] / sigma_nodes[-2]
            else:
                factor = sigma_node_factor

            # Limit requested sigma values to 2 increments below min effective sigma
            # This will ensure that any sigma values well below min_sigma will not be filtered, while those
            # close to min_sigma will receive mixed-lengthscale filtering as intended
            sigma[sigma < min_sigma / (factor ** 2)] = min_sigma / (factor ** 2)

            # Insert as many sigma values as needed to get to lowest requested value (max 2 inserts)
            while sigma_nodes[0] > np.min(sigma) * 1.001:
                sigma_nodes = np.insert(sigma_nodes, 0, sigma_nodes[0] / factor)

        if len(sigma_nodes) > 1:
            node_delta = np.log(sigma_nodes[-1] / sigma_nodes[-2])
        else:
            node_delta = 1

        def get_node_weights(x):
            # Tile x and nodes to same shape with extra axis
            tile_shape = np.ones(np.ndim(x) + 1, dtype=int)
            tile_shape[0] = len(sigma_nodes)
            x_tile = np.tile(x, tile_shape)
            node_tile = np.tile(sigma_nodes, (*x.shape, 1))
            node_tile = np.moveaxis(node_tile, -1, 0)

            nw = np.abs(np.log(x_tile / node_tile)) / node_delta
            nw[nw >= 1] = 1
            nw = 1 - nw
            return nw

        node_outputs = np.empty((len(sigma_nodes), *a.shape), dtype=a.dtype)

        for i in range(len(sigma_nodes)):
            if sigma_nodes[i] < min_sigma:
                # Sigma is below minimum effective value
                if empty:
                    # For empty filter, still need to apply filter to determine central value
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=min_sigma, axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate, order=order)
                else:
                    # For standard filter, reduces to original array
                    node_outputs[i] = a
            else:
                if empty:
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate, order=order)
                else:
                    node_outputs[i] = ndimage.gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode,
                                                                cval=cval,
                                                                truncate=truncate, order=order)

        node_weights = get_node_weights(sigma)
        out = node_outputs * node_weights
        return np.sum(out, axis=0)

    else:
        # No filtering to perform on this axis
        return a
