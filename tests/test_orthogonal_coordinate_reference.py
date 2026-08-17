import math

import torch

from src.orthogonal import EmpiricalOrthogonalBasis, solve_transported_history


DTYPE = torch.float64
RTOL = 1e-11
ATOL = 1e-12


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def _empirical_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.square()))


def _factor_history(
    past: torch.Tensor,
    current: torch.Tensor,
    *,
    eps_rank: float = 1e-12,
):
    coordinates = EmpiricalOrthogonalBasis(
        particle_count=past.shape[0],
        eps_rank=eps_rank,
        dtype=past.dtype,
        device=past.device,
    )
    for column in past.T:
        coordinates.project_and_update(column)

    past_basis = coordinates.B.clone()
    past_theta = coordinates.Theta.clone()
    current_step = coordinates.project_and_update(current)
    return coordinates, past_basis, past_theta, current_step


def _evaluate_q_p_fixture(
    W: torch.Tensor,
    G: torch.Tensor,
    Q_past: torch.Tensor,
    P_past: torch.Tensor,
    z_g: torch.Tensor,
    z_w: torch.Tensor,
    *,
    delta_bar: float,
):
    K_w, K_g = W.shape[0], G.shape[0]
    W_past, w = W[:, :-1], W[:, -1]
    G_past, g = G[:, :-1], G[:, -1]

    weight, B_w_past, Theta_w_past, weight_step = _factor_history(
        W_past, w
    )
    residual, B_g_past, Theta_g_past, residual_step = _factor_history(
        G_past, g
    )

    Q_w_past = solve_transported_history(
        Q_past, Theta_w_past, rcond=1e-14, driver="gelsd"
    )
    P_g_past = solve_transported_history(
        P_past, Theta_g_past, rcond=1e-14, driver="gelsd"
    )

    psi_w = (P_g_past.T @ weight_step.residual) / K_w
    q_coordinate = (
        Q_w_past @ weight_step.beta
        + (B_g_past @ psi_w) / math.sqrt(delta_bar)
        + weight_step.rho * z_g
    )

    # Respect the causal order: the q produced above is the current column of Q_t
    # used by the backward memory term.
    Q = torch.column_stack((Q_past, q_coordinate))
    Q_w = solve_transported_history(
        Q, weight.Theta, rcond=1e-14, driver="gelsd"
    )
    psi_g = (Q_w.T @ residual_step.residual) / K_g
    p_coordinate = (
        P_g_past @ residual_step.beta
        + math.sqrt(delta_bar) * (weight.B @ psi_g)
        + residual_step.rho * z_w
    )

    # Direct Moore--Penrose reference. These Gram matrices appear only in the
    # tests, never in the production orthogonal-coordinate implementation.
    C_w_past = (W_past.T @ W_past) / K_w
    alpha_w = torch.linalg.pinv(C_w_past) @ ((W_past.T @ w) / K_w)
    w_perp_direct = w - W_past @ alpha_w

    C_g_past = (G_past.T @ G_past) / K_g
    phi_w = torch.linalg.pinv(C_g_past) @ (
        (P_past.T @ w_perp_direct) / K_w
    )
    q_direct = (
        Q_past @ alpha_w
        + (G_past @ phi_w) / math.sqrt(delta_bar)
        + _empirical_norm(w_perp_direct) * z_g
    )

    alpha_g = torch.linalg.pinv(C_g_past) @ ((G_past.T @ g) / K_g)
    g_perp_direct = g - G_past @ alpha_g
    C_w = (W.T @ W) / K_w
    phi_g = torch.linalg.pinv(C_w) @ ((Q.T @ g_perp_direct) / K_g)
    p_direct = (
        P_past @ alpha_g
        + math.sqrt(delta_bar) * (W @ phi_g)
        + _empirical_norm(g_perp_direct) * z_w
    )

    return {
        "weight": weight,
        "residual": residual,
        "weight_step": weight_step,
        "residual_step": residual_step,
        "B_w_past": B_w_past,
        "B_g_past": B_g_past,
        "Theta_w_past": Theta_w_past,
        "Theta_g_past": Theta_g_past,
        "Q_w_past": Q_w_past,
        "P_g_past": P_g_past,
        "Q_w": Q_w,
        "psi_w": psi_w,
        "psi_g": psi_g,
        "alpha_w": alpha_w,
        "alpha_g": alpha_g,
        "phi_w": phi_w,
        "phi_g": phi_g,
        "w_perp_direct": w_perp_direct,
        "g_perp_direct": g_perp_direct,
        "q_coordinate": q_coordinate,
        "q_direct": q_direct,
        "p_coordinate": p_coordinate,
        "p_direct": p_direct,
        "Q": Q,
    }


def _shared_fluctuation_histories():
    Q_past = torch.tensor(
        [
            [0.2, -0.5],
            [1.0, 0.3],
            [-0.7, 0.8],
            [0.4, -1.1],
            [1.2, 0.0],
            [-0.3, 0.5],
            [0.6, -0.2],
        ],
        dtype=DTYPE,
    )
    P_past = torch.tensor(
        [
            [0.5, -0.2],
            [-1.0, 0.7],
            [0.3, 1.1],
            [0.8, -0.5],
            [-0.6, 0.4],
        ],
        dtype=DTYPE,
    )
    z_g = torch.tensor(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=DTYPE
    )
    z_w = torch.tensor([-0.3, 0.2, 0.4, -0.5, 0.1], dtype=DTYPE)
    return Q_past, P_past, z_g, z_w


def _assert_represented_vector_identities(
    result,
    W: torch.Tensor,
    G: torch.Tensor,
    Q_past: torch.Tensor,
    P_past: torch.Tensor,
) -> None:
    W_past = W[:, :-1]
    G_past = G[:, :-1]

    _assert_close(
        Q_past @ result["alpha_w"],
        result["Q_w_past"] @ result["weight_step"].beta,
    )
    _assert_close(
        G_past @ result["phi_w"],
        result["B_g_past"] @ result["psi_w"],
    )
    _assert_close(
        P_past @ result["alpha_g"],
        result["P_g_past"] @ result["residual_step"].beta,
    )
    _assert_close(
        W @ result["phi_g"],
        result["weight"].B @ result["psi_g"],
    )


def test_well_conditioned_q_p_match_direct_reference_with_unequal_populations():
    K_w, K_g = 5, 7
    B_w = math.sqrt(K_w) * torch.eye(K_w, dtype=DTYPE)[:, :3]
    B_g = math.sqrt(K_g) * torch.eye(K_g, dtype=DTYPE)[:, :3]
    Theta_w = torch.tensor(
        [[2.0, 0.5, -0.25], [0.0, 1.5, 0.4], [0.0, 0.0, 0.8]],
        dtype=DTYPE,
    )
    Theta_g = torch.tensor(
        [[1.2, -0.3, 0.6], [0.0, 1.1, -0.2], [0.0, 0.0, 0.9]],
        dtype=DTYPE,
    )
    W, G = B_w @ Theta_w, B_g @ Theta_g
    Q_past, P_past, z_g, z_w = _shared_fluctuation_histories()

    result = _evaluate_q_p_fixture(
        W, G, Q_past, P_past, z_g, z_w, delta_bar=2.25
    )

    assert result["q_coordinate"].shape == (K_g,)
    assert result["p_coordinate"].shape == (K_w,)
    _assert_close(result["weight"].B.T @ result["weight"].B / K_w, torch.eye(3, dtype=DTYPE))
    _assert_close(result["residual"].B.T @ result["residual"].B / K_g, torch.eye(3, dtype=DTYPE))
    _assert_close(W, result["weight"].B @ result["weight"].Theta)
    _assert_close(G, result["residual"].B @ result["residual"].Theta)
    _assert_close(result["weight_step"].residual, result["w_perp_direct"])
    _assert_close(result["residual_step"].residual, result["g_perp_direct"])
    _assert_represented_vector_identities(result, W, G, Q_past, P_past)
    _assert_close(result["q_coordinate"], result["q_direct"])
    _assert_close(result["p_coordinate"], result["p_direct"])

    _assert_close(
        result["q_coordinate"],
        torch.tensor(
            [
                0.06609546088265655,
                0.4024078783168049,
                0.5874999999999999,
                -0.6900000000000002,
                0.16999999999999998,
                -0.2891666666666666,
                0.3916666666666666,
            ],
            dtype=DTYPE,
        ),
    )
    _assert_close(
        result["p_coordinate"],
        torch.tensor(
            [
                -0.4056990217228608,
                0.33980467670609205,
                0.638651109528686,
                0.004545454545454408,
                -0.2554545454545454,
            ],
            dtype=DTYPE,
        ),
    )


def test_exact_rank_deficiency_matches_direct_moore_penrose_representations():
    K_w, K_g = 5, 7
    B_w = math.sqrt(K_w) * torch.eye(K_w, dtype=DTYPE)[:, :2]
    B_g = math.sqrt(K_g) * torch.eye(K_g, dtype=DTYPE)[:, :2]
    Theta_w = torch.tensor(
        [[1.0, 2.0, 0.5], [0.0, 0.0, 1.25]], dtype=DTYPE
    )
    Theta_g = torch.tensor(
        [[1.0, -3.0, -0.4], [0.0, 0.0, 0.75]], dtype=DTYPE
    )
    W, G = B_w @ Theta_w, B_g @ Theta_g
    Q_past, P_past, z_g, z_w = _shared_fluctuation_histories()

    result = _evaluate_q_p_fixture(
        W, G, Q_past, P_past, z_g, z_w, delta_bar=2.25
    )

    assert torch.linalg.matrix_rank(W[:, :-1]).item() == 1
    assert torch.linalg.matrix_rank(G[:, :-1]).item() == 1
    assert result["weight_step"].rank_before == 1
    assert result["weight_step"].rank_after == 2
    assert result["residual_step"].rank_before == 1
    assert result["residual_step"].rank_after == 2
    _assert_close(W, result["weight"].B @ result["weight"].Theta)
    _assert_close(G, result["residual"].B @ result["residual"].Theta)
    _assert_represented_vector_identities(result, W, G, Q_past, P_past)
    _assert_close(result["q_coordinate"], result["q_direct"])
    _assert_close(result["p_coordinate"], result["p_direct"])

    _assert_close(
        result["q_coordinate"],
        torch.tensor(
            [-0.26066412212681356, -0.09, 0.465, -0.68, 0.745, -0.68, 0.895],
            dtype=DTYPE,
        ),
    )
    _assert_close(
        result["p_coordinate"],
        torch.tensor(
            [0.0352555317022657, 0.08384029268608373, 0.42, -0.467, 0.147],
            dtype=DTYPE,
        ),
    )


def test_coordinate_solve_outperforms_normal_equations_when_ill_conditioned():
    K_g = 80
    epsilon = 1e-4
    Theta = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, epsilon, 2.0 * epsilon],
            [0.0, 0.0, epsilon**2],
        ],
        dtype=DTYPE,
    )
    beta = torch.tensor([0.7, -0.3, 0.2], dtype=DTYPE)
    indices = torch.arange(K_g * 3, dtype=DTYPE).reshape(K_g, 3)
    exact_transport = torch.sin(indices / 7.0) + 0.1 * torch.cos(indices / 11.0)
    Q = exact_transport @ Theta

    transported = solve_transported_history(
        Q, Theta, rcond=1e-14, driver="gelsd"
    )
    truth = exact_transport @ beta
    coordinate_term = transported @ beta
    coordinate_relative_error = torch.linalg.vector_norm(
        coordinate_term - truth
    ) / torch.linalg.vector_norm(truth)

    gram = Theta.T @ Theta
    cross = Theta.T @ beta

    assert torch.linalg.cond(Theta).item() > 1e8
    assert torch.linalg.cond(gram).item() > 1e15
    assert coordinate_relative_error.item() < 1e-6

    try:
        normal_coefficients = torch.linalg.solve(gram, cross)
    except torch.linalg.LinAlgError:
        # Failure to solve the nearly singular normal equations is itself the
        # expected instability; the coordinate solve above must still succeed.
        return

    normal_relative_error = torch.linalg.vector_norm(
        Q @ normal_coefficients - truth
    ) / torch.linalg.vector_norm(truth)
    assert normal_relative_error.item() > 1e-4
