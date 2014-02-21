import unittest
import numpy as np
from numpy.testing import assert_array_equal as assertAE

from rdp import rdp


class RDPTest(unittest.TestCase):
    def test_two(self):
        """
        Point sequence with only two elements.
        """
        assertAE(rdp(np.array([[0, 0], [4, 4]])),
                 np.array([[0, 0], [4, 4]]))

    def test_hor(self):
        """
        Horizontal line.
        """
        assertAE(rdp(np.array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]).reshape(5, 2)),
                 np.array([0, 0, 4, 0]).reshape(2, 2))

    def test_ver(self):
        """
        Vertical line.
        """
        assertAE(rdp(np.array([0, 0, 0, 1, 0, 2, 0, 3, 0, 4]).reshape(5, 2)),
                 np.array([0, 0, 0, 4]).reshape(2, 2))

    def test_diag(self):
        """
        Diagonal line.
        """
        assertAE(rdp(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape(5, 2)),
                 np.array([0, 0, 4, 4]).reshape(2, 2))

    def test_eps0(self):
        """
        Epsilon being to small to be simplified.
        """
        assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2)),
                 np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2))

    def test_eps1(self):
        """
        Epsilon large enough to be simplified.
        """
        assertAE(rdp(np.array([0, 0, 5, 1, 10, 1]).reshape(3, 2), 1),
                 np.array([0, 0, 10, 1]).reshape(2, 2))

    def test_L0(self):
        """
        Point sequence which has the form of a L.
        """
        assertAE(rdp(np.array([5, 0, 4, 0, 3, 0, 3, 1, 3, 2]).reshape(5, 2)),
                 np.array([5, 0, 3, 0, 3, 2]).reshape(3, 2))

    def test_nn(self):
        """
        Non-numpy interface to rdp.
        """
        self.assertEqual(rdp([[0, 0], [2, 2], [4, 4]]), [[0, 0], [4, 4]])
