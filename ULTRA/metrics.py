from numpy import random
import pingouin as pg
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import cohen_kappa_score, mean_squared_error, confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa

def icc(predicitions, targets):
    '''
    Calculate the intra-class correlation coefficient (ICC) metric between the predicted resuls and the targets.
    Predictions: list of the predicted Tumor Cellularities of the dataset
    targets:  list of the label Tumor Cellularities of the dataset
    '''
    d = {'targets': np.hstack([np.arange(1, len(predicitions) + 1, 1), np.arange(1, len(predicitions) + 1, 1)]),
                            'raters': np.hstack([np.tile(np.array(['M']), len(predicitions)),
                            np.tile(np.array(['A']), len(predicitions))]),
                            'scores': np.hstack([predicitions, targets])}
    df = pd.DataFrame(data=d)
    out_icc = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='scores')

    ci_95 = out_icc.loc[2,'CI95%']
    icc_score = out_icc.loc[2, "ICC"]
    CI_95 = f"[{ci_95[0]:.3f},{ci_95[1]:.3f}]"
    
    icc_score = round(icc_score, 3)

    return icc_score, CI_95


def pk(x, y, initial_lexsort=True):
    """
    Calculates the prediction probability. Adapted from scipy's implementation of Kendall's Tau

    Note: x should be the truth labels.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Whether to use lexsort or quicksort as the sorting method for the
        initial sort of the inputs. Default is lexsort (True), for which
        `predprob` is of complexity O(n log(n)). If False, the complexity is
        O(n^2), but with a smaller pre-factor (so quicksort may be faster for
        small arrays).
    Returns
    -------
    Prediction probability : float

    Notes
    -----
    The definition of prediction probability that is used is:
      p_k = (((P - Q) / (P + Q + T)) + 1)/2
    where P is the number of concordant pairs, Q the number of discordant
    pairs, and T the number of ties only in `y`.
    References
    ----------
    Smith W.D, Dutton R.C, Smith N.T. (1996) A measure of association for assessing prediction accuracy
    that is a generalization of non-parametric ROC area. Stat Med. Jun 15;15(11):1199-215
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if not x.size or not y.size:
        return (np.nan, np.nan)  # Return NaN if arrays are empty

    n = np.int64(len(x))
    temp = list(range(n))  # support structure used by mergesort
    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if y[perm[offs]] <= y[perm[offs+1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs+1]
            perm[offs+1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if y[perm[middle - 1]] < y[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and y[perm[offs + j]] <=
                                                y[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs+length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    if initial_lexsort:
        # sort implemented as mergesort, worst case: O(n log(n))
        perm = np.lexsort((y, x))
    else:
        # sort implemented as quicksort, 30% faster but with worst case: O(n^2)
        perm = list(range(n))
        perm.sort(key=lambda a: (x[a], y[a]))

    # compute joint ties
    first = 0
    t = 0
    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += ((i - first) * (i - first - 1)) // 2
            first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in x
    first = 0
    u = 0
    for i in range(1,n):
        if x[perm[first]] != x[perm[i]]:
            u += ((i - first) * (i - first - 1)) // 2
            first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges = mergesort(0, n)
    # compute ties in y after mergesort with counting
    first = 0
    v = 0
    for i in range(1,n):
        if y[perm[first]] != y[perm[i]]:
            v += ((i - first) * (i - first - 1)) // 2
            first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tot == u or tot == v:
        return (np.nan, np.nan)    # Special case for all ties in both ranks

    p_k = (((tot - (v + u - t)) - 2.0 * exchanges) / (tot - u) + 1)/2

    return p_k


def kappa_old(Predictions, targets):
    '''
    Calculate the kappa score metric between the predicted resuls and the targets.
    Predictions: list of the predicted Tumor Cellularities of the dataset
    targets:  list of the label Tumor Cellularities of the dataset
    '''
    # transfer discrete TC value to that of classes(Binning it into four categories of 0–25%, 26–50%,
    # 51–75%, and 76–100% to class 0,1,2,3.)
    if not isinstance(Predictions, np.ndarray):
        Predictions = np.array(Predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    pred_binned = pd.cut(Predictions, bins=[-0.0001,0.25,0.5,0.75,1], labels=False)
    target_binned = pd.cut(targets, bins=[-0.0001,0.25,0.5,0.75,1], labels=False)
    kappa_score = cohen_kappa_score(pred_binned, target_binned)
    return kappa_score


def mse(predictions, targets):
    '''
    Calculate the MSE metric between the predicted resuls and the targets.
    Predictions: list of the predicted Tumor Cellularities of the dataset
    targets:  list of the label Tumor Cellularities of the dataset
    '''
    # mse_score = mean_squared_error(predictions, targets)
    # Another way to calculate mse and its 95% confidence interval
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    error_squre = (predictions-targets)**2
    mse_score, lower_bound, upper_bound = confidence_interval(error_squre)
    CI_95 = f"[{lower_bound:.3f},{upper_bound:.3f}]"
    mse_score = round(mse_score, 3)
    return mse_score, CI_95


def confidence_interval(data, confidence=0.95):
    '''
    Calculate the confidence interval of a data sequence. 
    '''
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def kappa(Predictions, targets, confidence=0.95):
    '''
    Calculate the kappa score metric between the predicted resuls and the targets.
    Predictions: list of the predicted Tumor Cellularities of the dataset
    targets:  list of the label Tumor Cellularities of the dataset
    '''
    # transfer discrete TC value to that of classes(Binning it into four categories of 0–25%, 26–50%,
    # 51–75%, and 76–100% to class 0,1,2,3.)
    if not isinstance(Predictions, np.ndarray):
        Predictions = np.array(Predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    pred_binned = pd.cut(Predictions, bins=[-0.0001,0.25,0.5,0.75,1], labels=False)
    target_binned = pd.cut(targets, bins=[-0.0001,0.25,0.5,0.75,1], labels=False)
    N = len(targets)
    confusion = confusion_matrix(pred_binned, target_binned)
    out = cohens_kappa(confusion)
    k = out["kappa"]
    k_upper = out["kappa_upp"]
    k_low = out["kappa_low"]
    CI_95 = f"[{k_low:.3f},{k_upper:.3f}]"
    k = round(k, 3)
    return k, CI_95



# def confidence_interval1(data, confidence=0.95):
#     '''
#     Calculate the confidence interval of a data sequence. 
#     '''
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, std = np.mean(a), np.std(a)
#     t_crit = np.abs(scipy.stats.t.ppf((1-confidence)/2,n-1))
#     h = std*t_crit/np.sqrt(n)
#     return m, m-h, m+h


if __name__ == '__main__':
    pred = np.random.rand(100)

    ci1 = confidence_interval(pred)

    random.seed(1)
    targets = np.random.rand(100)
    icc_score, ci = icc(pred, targets)
    pkdd = pk(pred, targets)
    ddd = kappa(pred, targets)

    mes_score = mse(pred, targets)
    pass
