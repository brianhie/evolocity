# Evolocity

Evolocity is constructs protein sequence landscapes and uses language models to order sequences in this landscape in evolutionary time. Evolocity uses the local evolutionary predictions enabled by language models to enable global evolutionary insight.

Evolocity is a fork of the [scVelo](https://github.com/theislab/scvelo) tool for RNA velocity and relies on many aspects of the [Scanpy](https://scanpy.readthedocs.io/en/stable/) library for high-dimensional biological data analysis. Like Scanpy and scVelo, evolocity makes use of [anndata](https://anndata.readthedocs.io/en/latest/), an extremely convenient way to store and organizing biological data.

This repository also contains the analysis code and links to the data for the paper "Evolutionary velocity with protein language models" by Brian Hie, Kevin Yang, and Peter Kim.

## Documentation

In-depth API documentation is in progress.

## Installation

You should be able to install evolocity using `pip`:
```bash
python -m pip install evolocity
```

## Experiments

Below are scripts for reproducing the experiments in our paper. To apply evolocity to your own sequence data, we also encourage you to check out the tutorials in the documentation.

### Data

You can download the [relevant datasets](DATA_URL) (including training and validation data) using the commands
```bash
wget DATA_URL
tar xvf data.tar.gz
ln -s data/target/ target
```
within the same directory as this repository. Be sure to move the `target/` directory one level up.

### Dependencies

Before running the scripts below, we encourage you to use the [conda](https://docs.conda.io/en/latest/) environment in [environment-epi.yml](environment-epi.yml) using
```bash
conda env create --file environment-epi.yml
```
ESM-1b and TAPE need to be installed separately as described in [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm) and [https://github.com/songlab-cal/tape](https://github.com/songlab-cal/tape), respectively (PyTorch may need to be reupdated after TAPE installation).

Our experiments were run with Python version 3.7 on Ubuntu 20.04.

### Evolocity analysis

Our main evolocity analyses can be reproduced using the command
```bash
bash bin/main.sh
```
which will create new log files and figures in a new `figures/` directory.

### Scripts for other analyses

Phylogenetic tree reconstruction of ancient proteins can be done with the command
```bash
bash bin/phylo_eno.sh
bash bin/phylo_pgk.sh
bash bin/phylo_ser.sh
```

Deep mutational scan benchmarking can be done with the command
```bash
python bin/dms.py esm1b > dms_esm1b.log 2>&1
python bin/dms.py tape > dms_tape.log 2>&1
```