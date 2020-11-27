import pytest
import numpy as np
import sys
from search_eval.datasets import split_groups

def test_no_data():
    expected = (np.array([]), np.array([]), np.array([]), np.array([]))
    np.testing.assert_equal(expected, split_groups.split_test_train_groups(np.array([]), np.array([])))

def test_small_data():
    data =   np.array([1,2,3,4,5,6,7,8])
    groups = np.array([0,1,1,2,3,4,4,1]) #Must be started from 0!
    expected = (
                np.array([0, 4]), np.array([1, 5]), #train
                np.array([1, 2, 3, 5, 6, 7]), np.array([2, 3, 4, 6, 7, 8]) #test
                )
    np.testing.assert_equal(expected, split_groups.split_test_train_groups(data, groups, 0.6, 42))

def test_text_data():
    data = np.array([1,2,3,4,5,6,7])
    groups_text = np.array(['mirkur', 'mirkur2', 'mirkur', 'sobaka', 'sobaka2', 'sobaka sobaka', 'sobaka'])
    #encode groups
    classes, groups = np.unique(groups_text, return_inverse=True)
    np.testing.assert_equal(classes, np.array(['mirkur', 'mirkur2', 'sobaka', 'sobaka sobaka', 'sobaka2']))
    np.testing.assert_equal(groups, np.array([0, 1, 0, 2, 4, 3, 2]))
    expected = (
                np.array([0, 2, 3, 5, 6]), np.array([1, 3, 4, 6, 7]), #train
                np.array([1, 4]), np.array([2, 5]) #test
                )
    np.testing.assert_equal(expected, split_groups.split_test_train_groups(data, groups, 0.5, 42))