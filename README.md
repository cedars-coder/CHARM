# CHARM

CHARM is a Python implementation for detecting and locating changes in the
marginal distributions of high-dimensional time series. The method uses
full-sample ranks, convex-hull areas of local rank bridges, multiscale scanning,
and a dependent multiplier bootstrap.

The methodology is described in:

> Fu, X. and Hu, Y. *Nonparametric Change-Point Detection in High-Dimensional
> Time Series via Convex Hull Areas of Rank Bridges*.

## Requirements

CHARM requires Python 3.8 or later and the packages `NumPy` and `Numba`:

```bash
python -m pip install numpy numba
```

## Usage

```python
from CHARM import fit_charm

fit = fit_charm(X)
print(fit["change_points"])
```

`X` should be a finite `n x p` numeric array, with observations in rows and
variables in columns.

Main arguments include:

- `alpha`: significance level; default `0.05`;
- `n_boot`: number of bootstrap replications; default `199`;
- `seed`: bootstrap random seed;
- `windows`: optional half-window widths; the manuscript's geometric family is
  used by default;
- `numba_threads`: number of Numba threads; default `1`.

The returned object contains the estimated change points, multiscale detections,
test-statistic curves, bootstrap critical value, Monte Carlo p-value, and
diagnostic quantities. Change points are reported as split indices `b`, so the
corresponding data segments are `X[:b]` and `X[b:]`.

## Example

A small simulated example is provided in `Example.py`:

```bash
python Example.py
```

## Citation

If you use CHARM in academic work, please cite the accompanying paper. Citation
metadata are provided in `CITATION.cff`.

## Contact

For questions about the implementation, please contact **Xuesong Fu**.

## License

CHARM is available for non-commercial academic research, teaching, peer review,
replication, and methodological evaluation. See `LICENSE` for details.

