Ramer-Douglas-Peucker Algorithm
-------------------------------

.. image:: https://travis-ci.org/fhirschmann/rdp.png?branch=master
   :target: https://travis-ci.org/fhirschmann/rdp

.. image:: https://badge.fury.io/py/rdp.png
   :target: http://badge.fury.io/py/rdp

.. image:: https://readthedocs.org/projects/rdp/badge/?version=latest
   :target: http://rdp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Python/NumPy implementation of the Ramer-Douglas-Peucker algorithm
(Ramer 1972; Douglas and Peucker 1973) for 2D and 3D data.

The Ramer-Douglas-Peucker algorithm is an algorithm for reducing the number
of points in a curve that is approximated by a series of points.

Installation
````````````

.. code:: bash

    pip install rdp

Usage
`````

Simple pythonic interface:

.. code:: python

    from rdp import rdp

    rdp([[1, 1], [2, 2], [3, 3], [4, 4]])

.. code:: python

    [[1, 1], [4, 4]]

With epsilon=0.5:

.. code:: python

    rdp([[1, 1], [1, 1.1], [2, 2]], epsilon=0.5)

.. code:: python

    [[1.0, 1.0], [2.0, 2.0]]

Numpy interface:

.. code:: python

    import numpy as np
    from rdp import rdp

    rdp(np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2))

.. code:: python

    array([[1, 1],
           [4, 4]])

Links
`````

* `Documentation <http://rdp.readthedocs.io/en/latest/>`_
* `GitHub Page <http://github.com/fhirschmann/rdp>`_
* `PyPI <http://pypi.python.org/pypi/rdp>`_

References
``````````

Douglas, David H, and Thomas K Peucker. 1973. “Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or Its Caricature.” Cartographica: The International Journal for Geographic Information and Geovisualization 10 (2): 112–122.

Ramer, Urs. 1972. “An Iterative Procedure for the Polygonal Approximation of Plane Curves.” Computer Graphics and Image Processing 1 (3): 244–256.
