"""Module for split dataset to test and train
"""
import numpy as np
import numbers

def split_test_train_groups(data, groups, n_test_groups = 0.6, seed = None):
    """
    Splits <data> array into test/train with respect to groups array.
    Parameters
    ----------
    data : array-like
             An array of data elements
    groups : array-like
             An array of groups, first group must have index 0
    n_test_groups : int, float, optional
            Required number or fraction of groups in test dataset
    seed : int, optional
            Seed number to reproduce split next time
    Returns
    -------
    train index, train data, test index, test data : tuple of np.arrays
            
    """
    data = np.asarray(data)
    groups = np.asarray(groups)

    rng = np.random.RandomState(seed)

    classes, group_indices = np.unique(groups, return_inverse=True)
    #Calc split size
    if isinstance(n_test_groups, float):
        n_test_groups = int(n_test_groups * classes.shape[0])

    if not isinstance(n_test_groups, numbers.Integral): 
        raise Exception("n_test_groups must be int of float")

    classes_idx = np.arange(classes.shape[0], dtype=np.int64)
    permutation = rng.permutation(classes_idx)
    ind_test = permutation[:n_test_groups]
    ind_train = permutation[n_test_groups:]

    test_indx = np.flatnonzero(np.in1d(group_indices, ind_test))
    train_indx = np.flatnonzero(np.in1d(group_indices, ind_train))

    return train_indx, data[train_indx], test_indx, data[test_indx]