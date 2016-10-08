from __future__ import print_function

import time

import numpy as np
from numpy.testing import assert_array_equal as assertAE

import pytest

from rdp import rdp

nice_line = np.array([44, 95, 26, 91, 22, 90, 21, 90,
    19, 89, 17, 89, 15, 87, 15, 86, 16, 85,
    20, 83, 26, 81, 28, 80, 30, 79, 32, 74,
    32, 72, 33, 71, 34, 70, 38, 68, 43, 66,
    49, 64, 52, 63, 52, 62, 53, 59, 54, 57,
    56, 56, 57, 56, 58, 56, 59, 56, 60, 56,
    61, 55, 61, 55, 63, 55, 64, 55, 65, 54,
    67, 54, 68, 54, 76, 53, 82, 52, 84, 52,
    87, 51, 91, 51, 93, 51, 95, 51, 98, 50,
    105, 50, 113, 49, 120, 48, 127, 48, 131, 47,
    134, 47, 137, 47, 139, 47, 140, 47, 142, 47,
    145, 46, 148, 46, 152, 46, 154, 46, 155, 46,
    159, 46, 160, 46, 165, 46, 168, 46, 169, 45,
    171, 45, 173, 45, 176, 45, 182, 45, 190, 44,
    204, 43, 204, 43, 207, 43, 215, 40, 215, 38,
    215, 37, 200, 37, 195, 41]).reshape(77, 2)


@pytest.fixture
def line(length=100):
    arr = 5 * np.random.random_sample((length, 2))

    return arr.cumsum(0)

@pytest.fixture
def line3d(length=150):
    arr = 5 * np.random.random_sample((length, 3))

    return arr.cumsum(0)

@pytest.fixture
def inf():
    return float("inf")


def test_two():
    """
    Point sequence with only two elements.
    """
    assertAE(rdp(np.array([[0, 0], [4, 4]])),
             np.array([[0, 0], [4, 4]]))

def test_hor():
    """
    Horizontal line.
    """
    assertAE(rdp(np.array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]).reshape(5, 2)),
             np.array([0, 0, 4, 0]).reshape(2, 2))

def test_ver():
    """
    Vertical line.
    """
    assertAE(rdp(np.array([0, 0, 0, 1, 0, 2, 0, 3, 0, 4]).reshape(5, 2)),
             np.array([0, 0, 0, 4]).reshape(2, 2))

def test_diag():
    """
    Diagonal line.
    """
    assertAE(rdp(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape(5, 2)),
             np.array([0, 0, 4, 4]).reshape(2, 2))

def test_3d():
    """
    3 dimensions.
    """
    assertAE(rdp(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
                 .reshape(5, 3)),
             np.array([0, 0, 0, 4, 4, 4]).reshape(2, 3))

def test_eps0():
    """
    Epsilon being to small to be simplified.
    """
    assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2)),
             np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2))

def test_eps1():
    """
    Epsilon large enough to be simplified.
    """
    assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2), 1),
             np.array([0, 0, 10, 1]).reshape(2, 2))

def test_L0():
    """
    Point sequence which has the form of an L.
    """
    assertAE(rdp(np.array([5, 0, 4, 0, 3, 0, 3, 1, 3, 2]).reshape(5, 2)),
             np.array([5, 0, 3, 0, 3, 2]).reshape(3, 2))

def test_nn():
    """
    Non-numpy interface to rdp.
    """
    assert rdp([[0, 0], [2, 2], [4, 4]]) == [[0, 0], [4, 4]]

def test_rec_iter(line):
    for e in range(0, 10):
        assertAE(rdp(line, e * .1, algo="iter"), rdp(line, e * .1, algo="rec"))

def test_rec_iter3d(line3d):
    for e in range(0, 10):
        assertAE(rdp(line3d, e * .1, algo="iter"), rdp(line3d, e * .1, algo="rec"))

def test_inf_e(line, inf):
    res = rdp(line, inf)
    assert res.shape == (2, 2)

def test_inf_e_3d(line3d, inf):
    res = rdp(line3d, inf)
    assert res.shape == (2, 3)

def test_rec_iter2():
    for i in range(0, 40):
        assertAE(rdp(nice_line, algo="iter", epsilon=i * 0.1),
                 rdp(nice_line, algo="rec", epsilon=i * 0.1))
