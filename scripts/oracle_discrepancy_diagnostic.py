#!/usr/bin/env python3
"""Run the additive finite ST/oracle discrepancy diagnostic.

The default is a reduced-cost, balanced, fixed-zero-bias specialization of the
confidence-threshold experiment (seed 3100, delta=2, rho=.1, eta=.1,
lambda=.1, pi=5, kappa=logit(.8)).  Repeated trials, dimension sweeps, and
(pi,kappa) sweeps are explicit opt-ins and never run by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logit

from notebooks.experiment_helpers import make_algorithm_config, run_experiment
from notebooks.oracle_discrepancy_diagnostics import (
    format_oracle_discrepancy_report,
    plot_oracle_discrepancy_diagnostic,
    run_finite_oracle_discrepancy_diagnostic,
)


def _run_one(*, d: int, T: int, pi: float, kappa: float, seed: int, with_se: bool):
    config = make_algorithm_config(
        T=T,
        eta=0.1,
        penalty=0.1,
        pi=pi,
        kappa=kappa,
        include_bias=False,
        initial_bias=0.0,
    )
    kwargs = dict(
        name="finite ST/oracle discrepancy",
        d=d,
        delta=2.0,
        n_test=500,
        label_prior=0.5,
        rho=0.1,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=seed,
    )
    if with_se:
        kwargs.update(K_w=256, K_g=320)
    else:
        kwargs.update(run_state_evolution=False)
    run = run_experiment(**kwargs)
    return run_finite_oracle_discrepancy_diagnostic(
        run,
        grid_size=121,
        compute_state_evolution_lower_bound=with_se,
    )


def _record(diagnostic, **coordinates):
    def finite_or_none(value):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None

    gap = float(diagnostic.summary["oracle_gap"])
    bound = float(diagnostic.summary["oracle_gap_upper_bound"])
    return {
        **coordinates,
        "oracle_gap": gap,
        "oracle_gain": float(diagnostic.summary["oracle_gain"]),
        "self_training_gain": float(diagnostic.summary["self_training_gain"]),
        "oracle_gap_upper_bound": bound,
        "bound_to_positive_gap_ratio": bound / gap if gap > 0.0 else None,
        "all_inequalities_satisfied": bool(
            diagnostic.summary["all_finite_inequalities_satisfied"]
        ),
        "inequality_checks": diagnostic.summary["exact_checks"],
        "best_comparison_time": int(diagnostic.summary["best_comparison_time"]),
        "largest_loss_stage": diagnostic.summary["largest_loss_stage"],
        "stage_max_slack_ratio": {
            name: finite_or_none(value)
            for name, value in diagnostic.summary["stage_max_slack_ratio"].items()
        },
        "empirical_selection_rate_floor": float(
            diagnostic.constants["omega_lower_empirical_full_horizon"]
        ),
        "state_evolution_v_min": finite_or_none(
            diagnostic.constants["v_min_state_evolution_full_horizon"]
        ),
        "theoretical_selection_rate_floor": finite_or_none(
            diagnostic.constants["omega_lower_theoretical_full_horizon"]
        ),
        "log_theoretical_selection_rate_floor": finite_or_none(
            diagnostic.constants["log_omega_lower_theoretical_full_horizon"]
        ),
    }


def _dimension_plot(records: Iterable[dict]):
    records = list(records)
    d = np.asarray([item["d"] for item in records])
    gap = np.asarray([item["oracle_gap"] for item in records])
    bound = np.asarray([item["oracle_gap_upper_bound"] for item in records])
    ratio = np.asarray(
        [np.nan if item["bound_to_positive_gap_ratio"] is None else item["bound_to_positive_gap_ratio"] for item in records]
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(d, gap, marker="o")
    axes[0].set(title="Actual oracle gap", xlabel="dimension d", ylabel="gap")
    axes[1].plot(d, bound, marker="o")
    axes[1].set(title="Finite diagnostic bound", xlabel="dimension d", ylabel="upper bound")
    axes[2].plot(d, ratio, marker="o")
    axes[2].set(title="Bound / positive gap", xlabel="dimension d", ylabel="ratio")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
    return fig


def _parameter_plot(records: Iterable[dict], pis: list[float], kappas: list[float]):
    records = list(records)
    fields = ("oracle_gap", "oracle_gain", "oracle_gap_upper_bound")
    titles = ("Actual oracle gap", "Oracle gain", "Oracle-gap upper bound")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, field, title in zip(axes, fields, titles):
        values = np.full((len(kappas), len(pis)), np.nan)
        for record in records:
            i = kappas.index(record["kappa"])
            j = pis.index(record["pi"])
            values[i, j] = record[field]
        image = ax.imshow(values, origin="lower", aspect="auto")
        ax.set(
            title=title,
            xlabel=r"pseudo-label weight $\pi$",
            ylabel=r"confidence threshold $\kappa$",
            xticks=np.arange(len(pis)),
            xticklabels=[f"{value:g}" for value in pis],
            yticks=np.arange(len(kappas)),
            yticklabels=[f"{value:.3g}" for value in kappas],
        )
        fig.colorbar(image, ax=ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--without-state-evolution", action="store_true")
    parser.add_argument("--repeated-trials", action="store_true")
    parser.add_argument("--dimension-sweep", action="store_true")
    parser.add_argument("--parameter-sweep", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    canonical_pi = 5.0
    canonical_kappa = float(logit(0.8))
    diagnostic = _run_one(
        d=args.dimension,
        T=args.iterations,
        pi=canonical_pi,
        kappa=canonical_kappa,
        seed=args.seed,
        with_se=not args.without_state_evolution,
    )
    print(format_oracle_discrepancy_report(diagnostic))

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_oracle_discrepancy_diagnostic(
            diagnostic,
            title="Finite self-training/oracle discrepancy bound",
            show=False,
        )
        fig.savefig(output_dir / "canonical_trajectory_diagnostic.pdf", bbox_inches="tight")
        plt.close(fig)

    all_results = {"canonical": _record(diagnostic, d=args.dimension, seed=args.seed)}

    if args.repeated_trials:
        repeated = []
        for index in range(5):
            item = _run_one(
                d=args.dimension,
                T=args.iterations,
                pi=canonical_pi,
                kappa=canonical_kappa,
                seed=args.seed + 1000 * index,
                with_se=False,
            )
            repeated.append(_record(item, repetition=index, seed=args.seed + 1000 * index))
        all_results["repeated_trials"] = repeated

    if args.dimension_sweep:
        dimension_records = []
        dimensions = [50, 100, 200, 400]
        for d in dimensions:
            item = _run_one(
                d=d,
                T=args.iterations,
                pi=canonical_pi,
                kappa=canonical_kappa,
                seed=args.seed,
                with_se=False,
            )
            dimension_records.append(_record(item, d=d))
        all_results["dimension_sweep"] = dimension_records
        if output_dir is not None:
            fig = _dimension_plot(dimension_records)
            fig.savefig(output_dir / "dimension_sweep.pdf", bbox_inches="tight")
            plt.close(fig)

    if args.parameter_sweep:
        pis = [1.0, 2.0, 5.0]
        kappas = [float(logit(value)) for value in (0.7, 0.8, 0.9)]
        parameter_records = []
        for pi in pis:
            for kappa in kappas:
                item = _run_one(
                    d=args.dimension,
                    T=args.iterations,
                    pi=pi,
                    kappa=kappa,
                    seed=args.seed,
                    with_se=False,
                )
                parameter_records.append(_record(item, pi=pi, kappa=kappa))
        all_results["parameter_sweep"] = parameter_records
        if output_dir is not None:
            fig = _parameter_plot(parameter_records, pis, kappas)
            fig.savefig(output_dir / "parameter_sweep.pdf", bbox_inches="tight")
            plt.close(fig)

    if output_dir is not None:
        (output_dir / "diagnostic_summary.json").write_text(
            json.dumps(all_results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
