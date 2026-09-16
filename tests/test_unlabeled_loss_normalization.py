import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.objectives import LogisticLoss
from src.primitives import pseudo_residual, selection_rate


@pytest.mark.parametrize("unlabeled_selection", [[1., 0.], [1., 1.], [0., 0.]])
def test_pseudo_residual_unlabeled_normalization(unlabeled_selection):
    scores = torch.tensor([.2, -.7, 2., .1])
    Y = torch.tensor([1., -1., 1., -1.])
    Delta = torch.tensor([1., 1., 0., 0.])
    Yhat = torch.tensor([1., -1., -1., 1.])
    selection = torch.tensor([0., 0., *unlabeled_selection])
    omega = selection_rate(selection, Delta)
    loss = LogisticLoss()
    kwargs = dict(scores=scores, Y=Y, Delta=Delta, Yhat=Yhat,
                  selection=selection, omega=omega, pi=.7, eta=.1,
                  rho=.5, loss_function=loss)

    default = pseudo_residual(**kwargs)
    normalized = pseudo_residual(**kwargs, normalize_unlabeled_loss=True)
    unnormalized = pseudo_residual(**kwargs, normalize_unlabeled_loss=False)
    labeled = -.1 * (Delta / .5 * loss.gradient(scores, Y))
    unlabeled = (1 - Delta) * .7 / .5 * selection * loss.gradient(scores, Yhat)
    torch.testing.assert_close(unnormalized, -.1 * (Delta / .5 * loss.gradient(scores, Y) + unlabeled))
    assert torch.equal(default, normalized)
    assert torch.equal(normalized[Delta == 1], labeled[Delta == 1])
    assert torch.equal(unnormalized[Delta == 1], labeled[Delta == 1])
    assert torch.isfinite(normalized).all()
    assert torch.isfinite(unnormalized).all()

    if omega > 0:
        # Exact legacy arithmetic, including the order of multiplication.
        legacy = -.1 * (Delta / .5 * loss.gradient(scores, Y)
                        + (1 - Delta) * .7 / .5 * (selection / omega.item())
                        * loss.gradient(scores, Yhat))
        assert torch.equal(default, legacy)
        assert torch.count_nonzero(unnormalized - labeled) > 0
        torch.testing.assert_close(normalized - labeled, (unnormalized - labeled) / omega)
    else:
        assert torch.equal(normalized, labeled)
        assert torch.equal(unnormalized, labeled)
    if omega == 1:
        assert torch.equal(normalized, unnormalized)


def test_notebook13_unnormalized_cell_plots_its_own_run():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/13_temporal_error_attribution_demo.ipynb"
    cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]
             if cell["cell_type"] == "code"]
    setup = next(cell for cell in cells if "def polish_attribution(" in cell)
    namespace = {}
    exec("\n".join(line for line in setup.splitlines() if not line.startswith("%")), namespace)
    namespace["display"] = lambda figure: None
    parameters = namespace["TemporalExperimentParameters"](T=3, d=24, delta=3., rho=.4, kappa=.5)
    namespace["PARAMETERS"] = parameters
    baseline = namespace["run_temporal_experiment"](parameters)
    # No supervised run exists: this cell must be independently executable.
    exec(next(cell for cell in cells if cell.startswith("unnormalized_parameters =")), namespace)
    run = namespace["unnormalized_run"]
    assert baseline.algo_cfg.normalize_unlabeled_loss
    assert not run.algo_cfg.normalize_unlabeled_loss
    first = baseline.finite.update_records_[0]
    other = run.finite.update_records_[0]
    assert 0 < first.omega < 1
    assert torch.equal(first.scores, other.scores)
    assert torch.equal(first.omega, other.omega)
    labeled = -.5 * baseline.environment.Delta / baseline.environment.rho * LogisticLoss().gradient(first.scores, baseline.environment.Y)
    torch.testing.assert_close(first.g - labeled, (other.g - labeled) / first.omega)
    expected = namespace["population_components"](run)[0]
    plotted = namespace["unnormalized"].figures["attribution"].axes[0].lines[0].get_ydata()
    np.testing.assert_array_equal(plotted, expected)
    assert not np.allclose(expected, namespace["population_components"](baseline)[0])
