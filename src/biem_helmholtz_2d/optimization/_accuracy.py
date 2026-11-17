"""Convergence studies for scattered fields and shape derivatives."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import array_api_compat.numpy as np_compat
import orjson
from array_api.latest import Array
from array_api_compat import array_namespace
from ie_circle import NystromInterpolant, trapezoidal_quadrature
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

from biem_helmholtz_2d import accuracy
from biem_helmholtz_2d._acoustic import (
    near_field,
    plot_near_field,
    plot_near_field_prepare,
    scattering_dirichlet,
)
from biem_helmholtz_2d._incident import plane_wave, plane_wave_grad
from biem_helmholtz_2d._objective import field_objective, field_objective_derivative
from biem_helmholtz_2d.optimization._shape import ParameterShape


def example_accuracy(
    *,
    path: Path,
    centers: Array,
    k: float,
    alpha: float,
    eta: float,
    target: Array,
    weight_target: float,
    include_cross_section: bool,
    desired_total_field: complex,
    n_solve: Sequence[int],
    n_quad: Sequence[int],
    step: float,
    n_modes: int,
    seed: int,
    n_reference: int | None = None,
) -> None:
    """Study a scattered field and two shape perturbations on independent grids."""
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("centers must have shape (n_scatterers, 2) with n_scatterers > 0")
    if n_modes < 1 or step <= 0 or not math.isfinite(step):
        raise ValueError("n_modes and step must be positive and step must be finite")
    if not math.isfinite(weight_target) or weight_target < 0:
        raise ValueError("weight_target must be finite and nonnegative")
    if include_cross_section and desired_total_field != 0:
        raise ValueError("the cloaking objective minimizes the total field itself")
    steps = [step * 10**power for power in range(-3, 4)]
    if any(h <= 0 or not math.isfinite(h) for h in steps):
        raise ValueError("central-difference steps must be positive and finite")
    if not n_solve or not n_quad:
        raise ValueError("n_solve and n_quad must be nonempty")
    if n_reference is None:
        n_reference = (3 * max(*n_solve, *n_quad) + 1) // 2
    xp = array_namespace(centers, target)
    dtype, device = centers.dtype, centers.device
    if target.ndim != 2 or target.shape[0] == 0 or target.shape[1] != 2:
        raise ValueError("target must have shape (n_targets, 2) with n_targets > 0")
    if not xp.isdtype(target.dtype, "real floating") or not bool(xp.all(xp.isfinite(target))):
        raise ValueError("target must contain finite real floating-point coordinates")
    if bool(xp.any(xp.linalg.vector_norm(centers[None, :, :] - target[:, None, :], axis=-1) <= 2)):
        raise ValueError("each target must be more than 2 from every center")
    wave_number = xp.asarray(k, dtype=dtype, device=device)
    coupling_alpha = xp.asarray(alpha, dtype=dtype, device=device)
    coupling_eta = xp.asarray(eta, dtype=dtype, device=device)
    direction = xp.asarray([1, 0], dtype=dtype, device=device)
    incident = plane_wave(wave_number, direction)
    incident_grad = plane_wave_grad(wave_number, direction)
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  # Reproducible numerical samples, not security.

    def random_coefficients() -> Array:
        values = xp.reshape(
            xp.asarray(
                [rng.uniform(-1, 1) for _ in range(centers.shape[0] * n_modes * 2)],
                dtype=dtype,
                device=device,
            ),
            (centers.shape[0], n_modes, 2),
        )
        return values / xp.sum(xp.linalg.vector_norm(values, axis=-1), axis=-1)[:, None, None]

    def shapes(coefficients: Array, *, perturbation: bool) -> ParameterShape:
        constant = xp.full(
            (*coefficients.shape[:-2], 1), 0 if perturbation else 1, dtype=dtype, device=device
        )
        return ParameterShape(
            center=xp.zeros((*coefficients.shape[:-2], 2), dtype=dtype, device=device)
            if perturbation
            else centers,
            cos_coefs=xp.concat([constant, coefficients[..., 0]], axis=-1),
            sin_coefs=coefficients[..., 1],
        )

    coefficients = random_coefficients()
    highest = xp.broadcast_to(
        xp.stack(
            [
                xp.concat([
                    xp.zeros(n_modes - 1, dtype=dtype, device=device),
                    xp.ones(1, dtype=dtype, device=device),
                ]),
                xp.zeros(n_modes, dtype=dtype, device=device),
            ],
            axis=-1,
        ),
        coefficients.shape,
    )
    perturbations = xp.stack([highest, random_coefficients()])
    shape = shapes(coefficients, perturbation=False)
    h_shape = shapes(perturbations, perturbation=True)

    @cache
    def solve(n: int, /) -> NystromInterpolant:
        return scattering_dirichlet(
            k=wave_number,
            shape=shape,
            incident_field=incident,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n=n,
        )

    @cache
    def field(solve_n: int, quad_n: int, /) -> Array:
        return near_field(
            solve(solve_n),
            target,
            shape=shape,
            k=wave_number,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n=quad_n,
        )

    def derivative(solve_n: int, quad_n: int, /) -> Array:
        density = solve(solve_n)
        _, terms = field_objective(
            density,
            target=target,
            weight_target=weight_target,
            include_cross_section=include_cross_section,
            shape=shape,
            incident_field=incident,
            k=wave_number,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n_quad=quad_n,
            desired_total_field=desired_total_field,
        )
        t, _ = trapezoidal_quadrature(quad_n, xp=xp, device=device, dtype=dtype)
        _, directional = field_objective_derivative(
            density,
            terms,
            shape=shape,
            incident_gradient=incident_grad(shape.x(t)),
            k=wave_number,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n_solve=solve_n,
            n_quad=quad_n,
        )
        return directional(h_shape)

    field_reference, field_values = accuracy(
        field, n_solve=n_solve, n_quad=n_quad, n_reference=n_reference
    )
    derivative_reference, derivative_values = accuracy(
        derivative, n_solve=n_solve, n_quad=n_quad, n_reference=n_reference
    )

    def perturbed_objective(coefficients: Array, /) -> Array:
        boundary = shapes(coefficients, perturbation=False)
        density = scattering_dirichlet(
            k=wave_number,
            shape=boundary,
            incident_field=incident,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n=n_reference,
        )
        value, _ = field_objective(
            density,
            target=target,
            weight_target=weight_target,
            include_cross_section=include_cross_section,
            shape=boundary,
            incident_field=incident,
            k=wave_number,
            alpha=coupling_alpha,
            eta=coupling_eta,
            n_quad=n_reference,
            desired_total_field=desired_total_field,
        )
        return value

    central = xp.stack([
        xp.stack([
            (
                perturbed_objective(coefficients + h * perturbation)
                - perturbed_objective(coefficients - h * perturbation)
            )
            / (2 * h)
            for perturbation in perturbations
        ])
        for h in steps
    ])

    xlim = (
        min(float(xp.min(centers[:, 0])) - 2.5, float(xp.min(target[:, 0])) - 1),
        max(float(xp.max(centers[:, 0])) + 2.5, float(xp.max(target[:, 0])) + 1),
    )
    ylim = (
        min(float(xp.min(centers[:, 1])) - 3, float(xp.min(target[:, 1])) - 1),
        max(float(xp.max(centers[:, 1])) + 3, float(xp.max(target[:, 1])) + 1),
    )
    wave = plot_near_field_prepare(
        solve(n_reference),
        incident,
        xlim=xlim,
        ylim=ylim,
        shape=shape,
        k=wave_number,
        alpha=coupling_alpha,
        eta=coupling_eta,
        n=n_reference,
        n_plot=200,
        isin_shape_n_quad=max(n_reference, 4 * n_modes),
        isin_shape_tol=1e-5,
    )["utot"]["re"]
    t_plot = xp.linspace(0, 2 * xp.pi, 1000, dtype=dtype, device=device)

    def serialize(value: Array, /) -> Any:
        if xp.isdtype(value.dtype, "complex floating"):
            return {"real": serialize(xp.real(value)), "imag": serialize(xp.imag(value))}
        return np_compat.asarray(xp.asarray(value, device="cpu")).tolist()

    data = {
        "settings": {
            "n_reference": n_reference,
            "n_solve": list(n_solve),
            "n_quad": list(n_quad),
            "step": step,
            "seed": seed,
            "centers": serialize(centers),
            "coefficients": serialize(coefficients),
            "perturbation_coefficients": serialize(perturbations),
            "k": k,
            "alpha": alpha,
            "eta": eta,
            "target": serialize(target),
            "weight_target": weight_target,
            "include_cross_section": include_cross_section,
            "desired_total_field": {
                "real": desired_total_field.real,
                "imag": desired_total_field.imag,
            },
        },
        "field": {"reference": serialize(field_reference), "values": serialize(field_values)},
        "derivative": {
            "reference": serialize(derivative_reference),
            "values": serialize(derivative_values),
        },
        "central_difference": {"steps": steps, "values": serialize(central)},
        "total_field": {**wave, "data": serialize(wave["data"])},
        "boundaries": serialize(xp.moveaxis(shape.x(t_plot), -2, 0)),
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "accuracy.json").write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    example_accuracy_plot(path)


def example_accuracy_plot(path: Path, /) -> None:
    """Plot saved convergence studies and the field."""
    data = json.loads((path / "accuracy.json").read_text())
    if "wave" not in data and "total_field" not in data:
        _legacy_accuracy_plot(data, path)
        return
    settings = data["settings"]
    n_solve, n_quad = settings["n_solve"], settings["n_quad"]
    reference_label = f"reference: solve and integration $n={settings['n_reference']}$"

    def error_plot(values: Any, reference: Any, title: str, destination: Path) -> None:
        fig, ax = plt.subplots()
        for solve, row in zip(n_solve, values, strict=True):
            samples = sorted(zip(n_quad, row, strict=True), key=lambda sample: sample[0])
            orders, sampled_values = zip(*samples, strict=True)
            errors = np_compat.abs(np_compat.asarray(sampled_values) - reference)
            if errors.ndim > 1:
                errors = np_compat.max(errors, axis=tuple(range(1, errors.ndim)))
            ax.loglog(orders, np_compat.maximum(errors, 1e-300), "o-", label=f"n = {solve}")
        ticks = sorted(set(n_quad))
        ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Integration parameter $n$ ($2n-1$ nodes)")
        ax.set_ylabel("Absolute difference from reference")
        ax.set_title(f"{title}\n({reference_label})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Solve parameter")
        fig.tight_layout()
        fig.savefig(destination)
        plt.close(fig)

    reference = data["field"]["reference"]
    field_reference = np_compat.asarray(reference["real"]) + 1j * np_compat.asarray(
        reference["imag"]
    )
    field_values = data["field"]["values"]
    field_values = np_compat.asarray(field_values["real"]) + 1j * np_compat.asarray(
        field_values["imag"]
    )
    title = (
        "Maximum scattered-field difference over targets"
        if field_reference.size > 1
        else "Scattered-field difference at the target"
    )
    error_plot(field_values, field_reference, title, path / "scattered_field.svg")
    derivative_reference = np_compat.asarray(data["derivative"]["reference"])
    derivative_values = np_compat.asarray(data["derivative"]["values"])
    central = np_compat.asarray(data["central_difference"]["values"])
    cases = (("highest_mode", "highest Fourier mode"), ("random", "random perturbation"))
    for index, (case, case_label) in enumerate(cases):
        directory = path / case
        directory.mkdir(exist_ok=True)
        error_plot(
            derivative_values[..., index],
            derivative_reference[index],
            f"Objective-derivative difference: {case_label}",
            directory / "shape_derivative.svg",
        )
        fig, ax = plt.subplots()
        ax.loglog(
            data["central_difference"]["steps"],
            np_compat.maximum(
                np_compat.abs(central[:, index] - derivative_reference[index]), 1e-300
            ),
            "o-",
        )
        ax.set_xlabel(r"Finite-difference step size $h$")
        ax.set_ylabel("Absolute difference from reference adjoint")
        ax.set_title(f"Central-difference validation: {case_label}\n({reference_label})")
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        fig.tight_layout()
        fig.savefig(directory / "central_difference.svg")
        plt.close(fig)

    if "total_field" in data:
        fig, ax = plt.subplots()
        plot_near_field({"utot": {"re": data["total_field"]}}, ax_utot_re=ax)
        axes = [ax]
        filename = "total_field.svg"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_near_field(
            {"uscat": data["wave"]}, ax_uscat_re=axes[0], ax_uscat_im=axes[1], ax_uscat_abs=axes[2]
        )
        filename = "scattered_wave.svg"
    for ax in axes:
        for boundary in data["boundaries"]:
            ax.plot(
                [point[0] for point in boundary], [point[1] for point in boundary], color="black"
            )
        targets = settings["target"] if "target" in settings else [settings["target_point"]]
        if len(targets) > 1:
            points = np_compat.asarray(targets)
            ax.plot(points[:, 0], points[:, 1], ".", color="black", markersize=3)
        else:
            ax.plot(
                *targets[0],
                marker="x",
                markersize=4 * plt.rcParams["lines.markersize"],
                color="yellow",
            )
    fig.tight_layout()
    fig.savefig(path / filename)
    plt.close(fig)


def _legacy_accuracy_plot(data: dict[str, Any], path: Path, /) -> None:
    for row in data["records"]:
        if "n_quad" not in row:
            row["n_quad"] = row["n_quadrature"]
    settings = data["settings"]
    n_solve = settings["n_solve"] if "n_solve" in settings else settings["solve_orders"]
    n_quad = settings["n_quad"] if "n_quad" in settings else settings["quadrature_orders"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics = (
        ("field_error", "Scattered field"),
        ("field_derivative_error", "Field shape derivative (central)"),
        ("objective_derivative_error", "Objective shape derivative (adjoint)"),
    )
    for axis, (key, title) in zip(axes, metrics, strict=True):
        for solve in n_solve:
            records = sorted(
                (row for row in data["records"] if row["n_solve"] == solve),
                key=lambda row: row["n_quad"],
            )
            axis.loglog(
                [row["n_quad"] for row in records],
                [max(row[key], 1e-300) for row in records],
                "o-",
                label=f"solve n={solve}",
            )
        values = sorted(set(n_quad))
        axis.set_xticks(values, [str(value) for value in values])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Integration n (2n-1 nodes)")
        axis.set_ylabel("Absolute error vs. numerical reference")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path / "accuracy.svg")
    plt.close(fig)
