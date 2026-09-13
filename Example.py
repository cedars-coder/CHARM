"""Small reproducible CHARM example."""

import argparse

import numpy as np

from CHARM import fit_charm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small CHARM example.")
    parser.add_argument("--n-boot", type=int, default=99)
    parser.add_argument("--numba-threads", type=int, default=1)
    parser.add_argument("--validate-engine", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(20260212)
    n, p, true_change = 300, 10, 150
    X = rng.normal(size=(n, p))
    X[true_change:, :3] += 1.5

    fit = fit_charm(
        X,
        n_boot=args.n_boot,
        seed=20260212,
        numba_threads=args.numba_threads,
        validate_engine=args.validate_engine,
    )

    print(f"True split index: {true_change}")
    print(f"Detected split indices: {fit['change_points']}")
    print(f"Global critical value: {fit['critical_value']:.6f}")
    print(f"Monte Carlo p-value: {fit['p_value']:.6f}")


if __name__ == "__main__":
    main()

