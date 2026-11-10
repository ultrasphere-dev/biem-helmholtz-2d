# biem-helmholtz-2d

<p align="center">
  <a href="https://github.com/ultrasphere-dev/biem-helmholtz-2d/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/ultrasphere-dev/biem-helmholtz-2d/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://biem-helmholtz-2d.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/biem-helmholtz-2d.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/ultrasphere-dev/biem-helmholtz-2d">
    <img src="https://img.shields.io/codecov/c/github/ultrasphere-dev/biem-helmholtz-2d.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/biem-helmholtz-2d/">
    <img src="https://img.shields.io/pypi/v/biem-helmholtz-2d.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/biem-helmholtz-2d.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/biem-helmholtz-2d.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://biem-helmholtz-2d.readthedocs.io" target="_blank">https://biem-helmholtz-2d.readthedocs.io </a>

**Source Code**: <a href="https://github.com/ultrasphere-dev/biem-helmholtz-2d" target="_blank">https://github.com/ultrasphere-dev/biem-helmholtz-2d </a>

---

Acoustic scattering from analytic 2D shapes in NumPy / PyTorch (Kress 1991)

## Installation

Install this via pip (or your favourite package manager):

`pip install biem-helmholtz-2d`

## Optimization and quadrature accuracy

The example starts from two unit circles and compares several Sobolev metrics:

```sh
uv run biem-helmholtz-2d optimize
```

For NVIDIA CUDA, enter the supplied Nix development shell, install the CUDA
extra, and select the device:

```sh
nix develop
uv sync --extra cuda
uv run biem-helmholtz-2d optimize --device cuda
```

On CUDA, the real-argument Bessel functions of orders 0, 1, and 2 used by
these kernels run on the GPU through Triton and NVIDIA libdevice. Other Bessel
orders and complex arguments currently use SciPy on the CPU and copy their
results back to the original device. Thus complex wave numbers work, but do
not benefit from the fully GPU-resident Bessel path.

When running the full test suite with CUDA visible, the sphere-reference
package's Numba CUDA initialization can fail during collection. As a workaround,
`NUMBA_DISABLE_CUDA=1 uv run pytest` uses its CPU path while still exercising
this project's PyTorch CUDA tests. This disables Numba CUDA for that process.

To study the quadrature at the **initial circles** using the same wave number,
centers, incident wave, and target as `optimize`, run:

```sh
uv run biem-helmholtz-2d accuracy --n-solve 10 --n-solve 20 \
  --n-quadrature 10 --n-quadrature 20 --n-reference 40
uv run biem-helmholtz-2d plot-accuracy
```

Pass `--device cuda` to `accuracy` as well to run the study on the GPU. The
commands write `accuracy.json` and `accuracy.svg` beneath `accuracy/`. Each
order `n` means `2n-1` nodes: `--n-solve` controls both forward and adjoint
Nyström solves; `--n-quadrature` controls field and sensitivity integrations.
`optimize --n-quadrature N` also permits a different integration order. The
field and sensitivity quadrature rules are cached by order, namespace, dtype,
and device; quadratures constructed inside the `ie-circle` solver are separate.

The study compares the scattered field, its central-difference shape
derivative, and the adjoint objective derivative against a higher-order
numerical reference. It perturbs the first circle's first cosine coefficient.
For other geometries, perturbations, incident fields, and scalar wave
parameters, `biem_helmholtz_2d.accuracy()` runs the same study and returns a
JSON-compatible mapping. Its `target` is a desired **scattered** field; for a
desired total field, subtract the incident field at the observation point.
The JSON also records the central-difference objective derivative, which can
be compared with the adjoint. These errors are estimates, not bounds: increase
`--n-reference` and vary `--step` to check reference and finite-difference
convergence. When solve and integration grids differ, the continuous-adjoint
quadrature is not generally the exact derivative of the discrete Nyström solve;
the comparison measures that discrepancy too. Optimized shapes and points
closer to a boundary can require more quadrature than the initial circles.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ultrasphere-dev"><img src="https://avatars.githubusercontent.com/u/231439132?v=4?s=80" width="80px;" alt="ultrasphere-dev"/><br /><sub><b>ultrasphere-dev</b></sub></a><br /><a href="https://github.com/ultrasphere-dev/biem-helmholtz-2d/commits?author=ultrasphere-dev" title="Code">💻</a> <a href="#ideas-ultrasphere-dev" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ultrasphere-dev/biem-helmholtz-2d/commits?author=ultrasphere-dev" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
