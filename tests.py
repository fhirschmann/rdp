import unittest
import numpy as np
from numpy.testing import assert_array_equal as assertAE

from rdp import rdp, rdp_nn


class RDPTest(unittest.TestCase):
    def test_two(self):
        assertAE(rdp(np.array([[0, 0], [4, 4]])),
                 np.array([[0, 0], [4, 4]]))

    def test_hor(self):
        assertAE(rdp(np.array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]).reshape(5, 2)),
                 np.array([0, 0, 4, 0]).reshape(2, 2))

    def test_ver(self):
        assertAE(rdp(np.array([0, 0, 0, 1, 0, 2, 0, 3, 0, 4]).reshape(5, 2)),
                 np.array([0, 0, 0, 4]).reshape(2, 2))

    def test_diag(self):
        assertAE(rdp(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape(5, 2)),
                 np.array([0, 0, 4, 4]).reshape(2, 2))

    def test_eps0(self):
        assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2)),
                 np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2))

    def test_eps1(self):
        assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2), 1),
                 np.array([0, 0, 10, 1]).reshape(2, 2))

    def test_nn(self):
        self.assertEqual(rdp_nn([[0, 0], [2, 2], [4, 4]]), [[0, 0], [4, 4]])
