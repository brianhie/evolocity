Datasets
=========

Ageing *Drosophila* brain
~~~~~~~~~~~~~~~~~~~~~~

This is sourced from `Davie et al.`_ (*Cell* 2018, `GSE 107451`_) and contains scRNA-seq data from a collection of fly brain cells along with each cell's age (in days). It is a useful dataset for exploring a common scenario in multi-modal integration: scRNA-seq data aligned to a 1-dimensional secondary modality. Please see the `example in Visualization`_ where this dataset is used. 

.. code-block:: Python

   import schema
   adata = schema.datasets.fly_brain()


Paired RNA-seq and ATAC-seq from mouse kidney cells
~~~~~~~~~~~~~~~~~~~~~~

This is sourced from `Cao et al.`_ (*Science* 2018, `GSE 117089`_) and contains paired RNA-seq and ATAC-seq data from a collection of mouse kidney cells. The AnnData object provided here has some additional processing done to remove very low count genes and peaks. This is a useful dataset for the case where one of the modalities is very sparse (here, ATAC-seq). Please see the example in `Paired RNA-seq and ATAC-seq`_ where this dataset is used. 

.. code-block:: Python

   import schema
   adata = schema.datasets.scicar_mouse_kidney()
   




.. _Davie et al.: https://doi.org/10.1016/j.cell.2018.05.057
.. _GSE 107451: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107451
.. _example in Visualization: https://schema-multimodal.readthedocs.io/en/latest/visualization/index.html#ageing-fly-brain
.. _Cao et al.: https://doi.org/10.1126/science.aau0730
.. _GSE 117089: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117089
.. _Paired RNA-seq and ATAC-seq: https://schema-multimodal.readthedocs.io/en/latest/recipes/index.html#paired-rna-seq-and-atac-seq
