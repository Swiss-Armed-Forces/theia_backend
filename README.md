![Logo of Theia](doc/source/_static/logo.svg)

## Installation

First, [install poetry](https://python-poetry.org/docs/#installation).

Then, the repository can be installed as follows:

```bash
poetry install
```

## Running unit tests

```bash
python -m unittest discover tests
```

## Running benchmarks

Execute the notebook ``benchmarks/benchmark.ipynb``.

Cost of running a single monostatic radar detection is approx. 400 μs, most of which
(approx. 320 μs) is due to the line-of-sight check.
