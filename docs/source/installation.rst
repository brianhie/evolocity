Installation
============

We recommend Python v3.6 or higher.

PyPI, Virtualenv, or Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install evolocity from PyPI_ using::

    pip install -U evolocity

where ``-U`` is short for ``--upgrade``.
If you get a ``Permission denied`` error, use ``pip install -U evolocity --user`` instead.

Dependencies
^^^^^^^^^^^^

- `anndata <https://anndata.readthedocs.io/>`_ - annotated data object.
- `scanpy <https://scanpy.readthedocs.io/>`_ - toolkit for high-dimensional data analysis.
- `numpy <https://docs.scipy.org/>`_, `scipy <https://docs.scipy.org/>`_, `pandas <https://pandas.pydata.org/>`_, `scikit-learn <https://scikit-learn.org/>`_, `matplotlib <https://matplotlib.org/>`_.


Using fast neighbor search via `hnswlib <https://github.com/nmslib/hnswlib>`_ further requires (optional)::

    pip install pybind11 hnswlib


.. _PyPI: https://pypi.org/project/scvelo

.. _Scanpy: http://scanpy.readthedocs.io

.. _anndata: https://anndata.readthedocs.io
