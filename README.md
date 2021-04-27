# Evolutionary velocity with protein language models

This repository contains the analysis code, links to the data, and pretrained models for the paper "Evolutionary velocity with protein language models" by Brian Hie, Kevin Yang, and Peter Kim.

### Documentation


### Installation

You should be able to instal evolocity using `pip`:
```
python -m pip install evolocity
```

You can also install the development version directly from the repository:
```
git clone https://github.com/brianhie/evolocity.git
cd evolocity/
python setup.py install
```

### Data

You can download the [relevant datasets](DATA_URL) (including training and validation data) using the commands
```bash
wget DATA_URL
tar xvf data.tar.gz
```
within the same directory as this repository.

### Dependencies

The major Python package requirements and their tested versions are in [requirements.txt](requirements.txt).

Our experiments were run with Python version 3.7 on Ubuntu 18.04.

### Experiments
