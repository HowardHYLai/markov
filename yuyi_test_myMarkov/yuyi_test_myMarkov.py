# -*- coding: utf-8 -*-

import numpy as np

def _digitize_1d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in range(n_samples):
        X_digit[i] = np.searchsorted(bins, X[i], side='left')
    raise Exception("hi")
    return X_digit


def _digitize_2d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in range(n_samples):
        X_digit[i] = np.searchsorted(bins[i], X[i], side='left')
    return X_digit


def _digitize(X, bins):
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    return X_binned.astype('int64')


def _reshape_with_nan(X, n_samples, lengths, max_length):
    X_fill = np.full((n_samples, max_length), np.nan)
    for i in range(n_samples):
        X_fill[i, :lengths[i]] = X[i]
    return X_fill


def _compute_bins(X, n_samples, n_bins):
    bin_edges = np.percentile(
            X, np.linspace(0, 100, n_bins + 1)[1:-1], axis=1
        ).T
    mask = np.c_[
        ~np.isclose(0, np.diff(bin_edges, axis=1), rtol=0, atol=1e-8),
        np.full((n_samples, 1), True)
    ]
    if (n_bins > 2) and np.any(~mask):
        samples = np.where(np.any(~mask, axis=0))[0]
        lengths = np.sum(mask, axis=1)
        max_length = np.max(lengths)

        bin_edges_ = list()
        for i in range(n_samples):
            bin_edges_.append(bin_edges[i][mask[i]])

        bin_edges = _reshape_with_nan(bin_edges_, n_samples,
                                      lengths, max_length)
    return bin_edges



def _digitize(X, bins):
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    return X_binned.astype('int64')


def KBinsDiscretizer_transform(X, n_samples, n_bins):
    """Bin the data.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_timestamps)
        Data to transform.

    Returns
    -------
    X_new : array-like, shape = (n_samples, n_timestamps)
        Binned data.

    """

    bin_edges = _compute_bins(X, n_samples, n_bins)
    X_new = _digitize(X, bin_edges)
    return X_new

# =============================================================================
# 
# =============================================================================

def _markov_transition_matrix(X_binned, n_samples, n_timestamps, n_bins):
    X_mtm = np.zeros((n_samples, n_bins, n_bins))
    for i in range(n_samples):
        for j in range(n_timestamps - 1):
            X_mtm[i, X_binned[i, j], X_binned[i, j + 1]] += 1
    return X_mtm


def _markov_transition_field(
    X_binned, X_mtm, n_samples, n_timestamps, n_bins
):
    X_mtf = np.zeros((n_samples, n_timestamps, n_timestamps))
    for i in range(n_samples):
        for j in range(n_timestamps):
            for k in range(n_timestamps):
                X_mtf[i, j, k] = X_mtm[i, X_binned[i, j], X_binned[i, k]]
    return X_mtf


def _aggregated_markov_transition_field(X_mtf, n_samples, image_size,
                                        start, end):
    X_amtf = np.empty((n_samples, image_size, image_size))
    for i in range(n_samples):
        for j in range(image_size):
            for k in range(image_size):
                X_amtf[i, j, k] = np.mean(
                    X_mtf[i, start[j]:end[j], start[k]:end[k]]
                )
    return X_amtf


# =============================================================================
# 
# =============================================================================
def segmentation(ts_size, window_size, overlapping=False, n_segments=None):
    """Compute the indices for Piecewise Agrgegate Approximation.

    Parameters
    ----------
    ts_size : int
        The size of the time series.

    window_size : int
        The size of the window.

    overlapping : bool (default = False)
        If True, overlapping windows may be used. If False, non-overlapping
        are used.

    n_segments : int or None (default = None)
        The number of windows. If None, the number is automatically
        computed using ``window_size``.

    Returns
    -------
    start : array
        The lower bound for each window.

    end : array
        The upper bound for each window.

    size : int
        The size of ``start``.

    Examples
    --------
    >>> from pyts.utils import segmentation
    >>> start, end, size = segmentation(ts_size=12, window_size=3)
    >>> print(start)
    [0 3 6 9]
    >>> print(end)
    [ 3  6  9 12]
    >>> size
    4

    """
    if not isinstance(ts_size, (int, np.integer)):
        raise TypeError("'ts_size' must be an integer.")
    if not ts_size >= 2:
        raise ValueError("'ts_size' must be an integer greater than or equal "
                         "to 2 (got {0}).".format(ts_size))
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("'window_size' must be an integer.")
    if not window_size >= 1:
        raise ValueError("'window_size' must be an integer greater than or "
                         "equal to 1 (got {0}).".format(window_size))
    if not window_size <= ts_size:
        raise ValueError("'window_size' must be lower than or equal to "
                         "'ts_size' ({0} > {1}).".format(window_size, ts_size))
    if not (n_segments is None or isinstance(n_segments, (int, np.integer))):
        raise TypeError("'n_segments' must be None or an integer.")
    if isinstance(n_segments, (int, np.integer)):
        if not n_segments >= 2:
            raise ValueError(
                "If 'n_segments' is an integer, it must be greater than or "
                "equal to 2 (got {0}).".format(n_segments)
            )
        if not n_segments <= ts_size:
            raise ValueError(
                "If 'n_segments' is an integer, it must be lower than or "
                "equal to 'ts_size' ({0} > {1}).".format(n_segments, ts_size)
            )

    if n_segments is None:
        quotient, remainder = divmod(ts_size, window_size)
        n_segments = quotient if remainder == 0 else quotient + 1

    if not overlapping:
        bounds = np.linspace(0, ts_size, n_segments + 1).astype('int64')
        start = bounds[:-1]
        end = bounds[1:]
        size = start.size
        return start, end, size
    else:
        n_overlapping = (n_segments * window_size) - ts_size
        n_overlaps = n_segments - 1
        overlaps = np.linspace(0, n_overlapping,
                               n_overlaps + 1).astype('int64')
        bounds = np.arange(0, (n_segments + 1) * window_size, window_size)
        start = bounds[:-1] - overlaps
        end = bounds[1:] - overlaps
        size = start.size
        return start, end, size
    
def MTF(X, n_bins, image_size=32):
    """Transform each time series into a MTF image.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_timestamps)
        Input data

    Returns
    -------
    X_new : array-like, shape = (n_samples, image_size, image_size)
        Transformed data. If ``flatten=True``, the shape is
        `(n_samples, image_size * image_size)`.

    """
    n_samples, n_timestamps = X.shape ## n_samples: 6540, n_timestamps: 32

    X_binned = KBinsDiscretizer_transform(X, n_samples, n_bins) ## X_binned.shape: (6540, 32)
    

    X_mtm = _markov_transition_matrix(X_binned, n_samples, n_timestamps, n_bins) ## X_mtm.shape: (6540, 8, 8)
    sum_mtm = X_mtm.sum(axis=2) ## sum_mtm.shape: (6540, 8)
    # raise Exception(X_mtm.sum(axis=2).shape)
    np.place(sum_mtm, sum_mtm == 0, 1)
    X_mtm /= sum_mtm[:, :, None]

    X_mtf = _markov_transition_field(
        X_binned, X_mtm, n_samples, n_timestamps, n_bins
    )

    window_size, remainder = divmod(n_timestamps, image_size)
    if remainder == 0:
        X_amtf = np.reshape(
            X_mtf, (n_samples, image_size, window_size,
                    image_size, window_size)
        ).mean(axis=(2, 4))
    else:
        window_size += 1
        start, end, _ = segmentation(
            n_timestamps, window_size, False, image_size
        )
        X_amtf = _aggregated_markov_transition_field(
            X_mtf, n_samples, image_size, start, end
        )

    return X_amtf



def main():
    from pyts.datasets import load_gunpoint
    from pyts.image import MarkovTransitionField
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd 
    import copy
    from fun_FITPS import FITPS
    
    V, I = None, None
    
    path = r"C:\Users\Yuyi\OneDrive\桌面\markov_fitps/0001Hairdry_vi.csv"
    data = pd.read_csv(path)
    V = list(data.iloc[:,0])
    I = list(data.iloc[:,1])
    
    
    ## 電壓過零點 
    zeros = []
    size = len(V)
    for i in range(size-1):
        if V[i] < 0 and V[i+1]>=0:
            zeros.append(i+1)
    
    ## fitps (你只要完成這部分就可以咯，不會可以問沂偉)
    ##########################################################################
    V_fitps = np.array([])
    I_fitps = np.array([])
    
    size = len(zeros)
    ## 以一秒為單位做FITPS轉換
    for i in range(0, size-60, 60):
        start, end = zeros[i], zeros[i+60]
        waves = np.array([V[start:end], I[start:end]]).T 
        waves_fitps = FITPS(waves, 32)  ## 這行要自己來唷，不能執行唷
        V_fitps = np.append(V_fitps, waves_fitps[:,0])
        I_fitps = np.append(I_fitps, waves_fitps[:,1])
    
    print("V_fitps長度 : ", len(V_fitps))
    print("I_fitps長度 : ", len(I_fitps))
    
    ##########################################################################
    
    ## 處理資料
    x = []
    size = len(V_fitps)
    for i in range(0, size, 32):
        start, end = i, i+32
        
        x.append(I_fitps[start:end])
    
    
    ## Markov
    X = np.array(x) ## shape: (6817, 32)  記得第二維不能有 32、33這樣跳，要統一大小    
    #############################################################
    #############################################################
    X = np.array([X[2870]])
    output = MTF(X, n_bins=8)
    
    # plt.imshow(output[2870])
    plt.imshow(output[0])
    pass

if __name__ == "__main__":
    main()
