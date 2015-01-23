from os import path
from setuptools import setup


setup(
    name="rdp",
    version="0.5",
    description="Pure Python implementation of the Ramer-Douglas-Peucker algorithm",
    long_description=open(path.join(path.dirname(__file__), "README.rst"), encoding="utf8").read(),
    url="http://github.com/fhirschmann/rdp",
    author="Fabian Hirschmann",
    author_email="fabian@hirschmann.email",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
    ],
    install_requires=[
        "numpy",
    ],
    platforms="any",
    keywords="rdp ramer douglas peucker line simplification numpy",
    packages=["rdp"],
)
