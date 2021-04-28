Installation
============

We recommend Python v3.6 or higher.

PyPI, Virtualenv, or Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``pip`` (or ``pip3``):

.. code-block:: bash

    pip install evolocity

Evolocity has been designed to be compatible with the popular and excellent single-cell Python package, Scanpy_ and relies on Scanpy's underlying data abstraction, anndata_.
We recommend installing the Docker image recommended_ by Scanpy maintainers and then using ``pip``, as described above, to install Schema in it.


.. _Scanpy: http://scanpy.readthedocs.io

.. _anndata: https://anndata.readthedocs.io
