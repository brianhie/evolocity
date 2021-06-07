.. automodule:: evolocity

API
===

Import evolocity as::

   import evolocity as evo


After reading the data (``evo.pp.featurize_fasta``) or loading an in-built dataset (``evo.datasets.*``),
the typical workflow consists of subsequent calls of
preprocessing (``evo.pp.*``), analysis tools (``evo.tl.*``), and plotting (``evo.pl.*``).


Preprocessing (pp)
------------------

**Featurization** (language model embedding)

.. autosummary::
   :toctree: .

   pp.featurize_seqs
   pp.featurize_fasta

**Landscape** (nearest neighbors graph construction)

.. autosummary::
   :toctree: .

   pp.neighbors


Tools (tl)
----------

**Velocity estimation**

.. autosummary::
   :toctree: .

   tl.velocity_graph
   tl.velocity_embedding

**Pseudotime and trajectory inference**

.. autosummary::
   :toctree: .

   tl.terminal_states
   tl.velocity_pseudotime

**Interpretation**

.. autosummary::
   :toctree: .

   tl.onehot_msa
   tl.residue_scores
   tl.random_walk

Plotting (pl)
-------------

Also see `scanpy's plotting API <https://scanpy.readthedocs.io/en/stable/api/scanpy.plotting.html>`_ for additional visualization functionality, including UMAP scatter plots.

**Velocity embeddings**

.. autosummary::
   :toctree: .

   pl.velocity_embedding
   pl.velocity_embedding_grid
   pl.velocity_embedding_stream
   pl.velocity_contour

**Mutation interpretation**

.. autosummary::
   :toctree: .

   pl.residue_scores
   pl.residue_categories


Datasets
--------

.. autosummary::
   :toctree: .

   datasets.nucleoprotein
   datasets.cytochrome_c


Settings
--------

.. autosummary::
   :toctree: .

   set_figure_params
