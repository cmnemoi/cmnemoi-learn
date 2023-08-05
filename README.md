# cmnemoi-learn - Machine Learning from scratch by Charles-Meldhine Madi Mnemoi

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