Overview
========


Schema is a general algorithm for integrating heterogeneous data
modalities. While it has been specially designed for multi-modal
single-cell biological datasets, it should work in other multi-modal
contexts too.

.. image:: ../_static/Schema-Overview-v2.png
   :width: 648
   :alt: 'Overview of Schema'
 
Schema is designed for single-cell assays where multiple modalities have
been *simultaneously* measured for each cell. For example, this could be
simultaneously-asayed ("paired") scRNA-seq and scATAC-seq data, or a
spatial-transcriptomics dataset (e.g. 10x Visium, Slideseq or
STARmap). Schema can also be used with just a scRNA-seq dataset where some
per-cell metadata is available (e.g., cell age, donor information, batch
ID etc.). With this data, Schema can help perform analyses like:

  * Characterize cells that look similar transcriptionally but differ
    epigenetically.

  * Improve cell-type inference by combining RNA-seq and ATAC-seq data.

  * In spatially-resolved single-cell data, identify differentially
    expressed genes (DEGs) specific to a spatial pattern.

  * **Improved visualizations**: tune t-SNE or UMAP plots to more clearly
    arrange cells along a desired manifold. 

  * Simultaneously account for batch effects while also integrating
    other modalities.

Intuition
~~~~~~~~~

To integrate multi-modal data, Schema takes a `metric learning`_
approach. Each modality is interepreted as a multi-dimensional space, with
observations mapped to points in it (**B** in figure above). We associate
a distance metric with each modality: the metric reflects what it means
for cells to be similar under that modality. For example, Euclidean
distances between L2-normalized expression vectors are a proxy for
coexpression. Across the three graphs in the figure (**B**), the dashed and
dotted lines indicate distances between the same pairs of
observations. 

Schema learns a new distance metric between points, informed
jointly by all the modalities. In Schema, we start by designating one
high-confidence modality as the *primary* (i.e., reference) and the
remaining modalities as *secondary*--- we've found scRNA-seq to typically
be a good choice for the primary modality.  Schema transforms the
primary-modality space by scaling each of its dimensions so that the
distances in the transformed space have a higher (or lower, if desired!)
correlation with corresponding distances in the secondary modalities
(**C,D** in the figure above). You can choose any distance metric for the
secondary modalities, though the primary modality's metric needs to be Euclidean.
The primary modality can be pre-transformed by
a `PCA`_ or `NMF`_ transformation so that the scaling occurs in this latter
space; this can often be more powerful because the major directions of variance are
now axis-aligned and hence can be scaled independently.

Advantages
~~~~~~~~~~

In generating a shared-space representation, Schema is similar to
statistical approaches like CCA (canonical correlation analysis) and 
deep-learning methods like autoencoders (which map multiple
representations into a shared latent space). Each of these approaches offers a
different set of trade-offs. Schema, for instance, requires the output
space to be a linear transformation of the primary modality. Doing so
allows it to offer the following advantages:

  * **Interpretability**: Schema identifies which features of the primary
    modality were important in maximizing its agreement with the secondary
    modalities. If the features corresponded to genes (or principal components),
    this can directly be interpreted in terms of gene importances. 

  * **Regularization**: single-cell data can be sparse and noisy. As we
    discuss in our `paper`_, unconstrained approaches like CCA and
    autoencoders seek to maximize the alignment between modalities without
    any other considerations. In doing so, they can pick up on artifacts
    rather than true biology. A key feature of Schema is its
    regularization: if enforces a limit on the distortion of the primary
    modality, making sure that the final result remains biologically
    informative.

  * **Speed and flexibility**: Schema is a based on a fast quadratic
    programming approach that allows for substantial flexibility in the
    number of secondary modalities supported and their relative weights. Also, arbitrary
    distance metrics (i.e., kernels) are supported for the secondary modalities.

    
Quick Start
~~~~~~~~~~~

Install via pip

.. code-block:: bash

    pip install schema_learn

**Example**: correlate gene expression with developmental stage. We demonstrate use with Anndata objects here.

.. code-block:: Python

    import schema
    adata = schema.datasets.fly_brain()  # adata has scRNA-seq data & cell age
    
    sqp = schema.SchemaQP( min_desired_corr=0.99, # require 99% agreement with original scRNA-seq distances
		           params= {'decomposition_model': 'nmf', 'num_top_components': 20} )
		    
    #correlate the gene expression with the 'age' parameter
    mod_X = sqp.fit_transform( adata.X, # primary modality
                               [ adata.obs['age'] ], # list of secondary modalities
			       [ 'numeric' ] )  # datatypes of secondary modalities
			       
    gene_wts = sqp.feature_weights() # get a ranking of gene wts important to the alignment


Paper & Code
~~~~~~~~~~~~

Schema is described in the paper *Schema: metric learning enables
interpretable synthesis of heterogeneous single-cell modalities*
(http://doi.org/10.1101/834549)

Source code available at: https://github.com/rs239/schema


.. _metric learning: https://en.wikipedia.org/wiki/Similarity_learning#Metric_learning
.. _paper: https://doi.org/10.1101/834549
.. _PCA: https://en.wikipedia.org/wiki/Principal_component_analysis
.. _NMF: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
