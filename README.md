# cmnemoi-learn - Machine Learning from scratch by Charles-Meldhine Madi Mnemoi

![CI Status](https://github.com/cmnemoi/cmnemoi-learn/actions/workflows/continous_integration.yaml/badge.svg?branch=main)
![CD Status](https://github.com/cmnemoi/cmnemoi-learn/actions/workflows/create_github_release.yaml/badge.svg?branch=main)
[![Coverage Status](https://coveralls.io/repos/github/cmnemoi/cmnemoi-learn/badge.svg?branch=main)](https://coveralls.io/github/cmnemoi/cmnemoi-learn?branch=main) 
[![PyPI version](https://badge.fury.io/py/cmnemoi-learn.svg)](https://badge.fury.io/py/cmnemoi-learn) 

Repository in which I will implement some of the machine learning models described in Elements Of Statistical Learning by Hastie, Tibshirani and Friedman from scratch (using only `numpy`) in form of a Python package.

The implementations will be unit tested against popular implementation (Scikit-learn, PyTorch) with `pytest`.

The quality of the code will be checked using `black`, `pylint` and `mypy` at each commit through a GitHub Action CI pipeline.

The package will be published on PyPI at each push to the `main` branch through a GitHub Action CD pipeline.

# Install the package

```bash
pip install cmnemoi-learn
```

# Contributing

Clone the repo :
```bash
git clone https://github.com/cmnemoi/cmnemoi-learn.git
cd cmnemoi-learn
```

Then install dependencies. If you run Miniconda or Anaconda: 
```bash
conda create -n cmnemoi-learn python=3.11 -y
conda activate cmnemoi-learn
pip install -r requirements.txt
```

If you run Poetry:
```bash
poetry install
```

# License

[MIT License](LICENSE.md)