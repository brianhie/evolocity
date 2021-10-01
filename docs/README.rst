=========
Evolocity
=========

Evolocity implements evolutionary velocity (evo-velocity), which models a protein sequence landscape as an evolutionary "vector field" by using the local evolutionary predictions enabled by language models to enable global evolutionary insight.

.. image:: ../cyc_stream.PNG
   :width: 450
   :alt: 'Evolocity overview'

Evo-velocity uses the change in languge model likelihoods to estimate directionality between two biological sequences.
Then, over an entire sequence similarity network, this procedure is used to direct network edges.
Finally, network diffusion analysis can identify roots, order sequences in pseudotime, and identify mutations driving the velocity.

Evolocity is a fork of the `scVelo <https://github.com/theislab/scvelo>`_ tool for RNA velocity and relies on many aspects of the `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_ library for high-dimensional biological data analysis.
Like Scanpy and scVelo, evolocity makes use of `anndata <https://anndata.readthedocs.io/en/latest/>`_, an extremely convenient way to store and organize biological data.


Quick Start
===========

Installation
------------

You should be able to install evolocity using ``pip``::

   python -m pip install evolocity

API example
-----------

Below is a quick Python example of using evolocity to load and analyze sequences in a FASTA file.

.. code-block:: python

   import evolocity as evo
   import scanpy as sc

   # Load sequences and compute language model embeddings.
   fasta_fname = 'data.fasta'
   adata = evo.pp.featurize_fasta(fasta_fname)

   # Construct sequence similarity network.
   evo.pp.neighbors(adata)

   # Run evolocity analysis.
   evo.tl.velocity_graph(adata)

   # Embed network and velocities in two-dimensions and plot.
   sc.tl.umap(adata)
   evo.tl.velocity_embedding(adata)
   evo.pl.velocity_embedding_grid(adata)
   evo.pl.velocity_embedding_stream(adata)
