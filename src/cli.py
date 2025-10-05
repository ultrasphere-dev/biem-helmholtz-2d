from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path

import numpy as np
import torch
import typer
from acoustic_obstacle_scattering.r2.main import get_uinf
from cm_time import timer
from rich.logging import RichHandler

# import tracemalloc


app = typer.Typer()

LOG = getLogger(__name__)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = INFO
    if verbose:
        level = DEBUG
    basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def farfield_showcase() -> None:
    N = 32
    t = torch.linspace(0, torch.pi * 2, 2 * N + 1)[:-1]
    directions = torch.stack([torch.cos(t), torch.sin(t)], dim=-1)
    result = get_uinf(1, N, wave_direction=(1, 0), farfield_direction=directions, eta=1)
    result_abs = torch.abs(result)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(t, result_abs, label="Amplitude")
    ax.plot(t, result.real, label="Real")
    # set title
    ax.set_title("Farfield pattern for a plane wave")
    # set x and y labels
    ax.set_xlabel("Theta (Radians)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    Path(".cache").mkdir(exist_ok=True)
    fig.savefig(".cache/farfield_showcase_2d.svg")
    fig.savefig(".cache/farfield_showcase_2d.png")


@app.command()
def benchmark_plot() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    dfs = {
        p.stem: pd.read_csv(
            p,
            names=[
                "n",
                "k",
                "device",
                "dtype",
                "time",
                "uinf",
                "max_memory_cached",
                "max_memory_allocated",
            ],
        )
        for p in Path("hpc/results").glob("*.csv")
    }
    for name, df in dfs.items():
        # df["machine"] = name
        df["machine"] = {
            "3950x_1060_6GB(16C)": "Computer 1 / 16C",
            "accms_56(56C)": "Laurel / 56C",
            "tsubame40_cpu_160(160C)": "TSUBAME 4.0 cpu_160 / 160C",
            "tsubame40_node_f(192C)": "TSUBAME 4.0 node_f / 192C",
            "tsubame40_node_q(48C)": "TSUBAME 4.0 node_q / 48C",
        }.get(name, name)
    df = pd.concat(dfs.values())
    # df.set_index("n", inplace=True)

    # fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="n", y="time", hue="device", markers=True, ax=ax)
    # df.plot(logx=True, logy=True, style="o",
    # title="Time vs n", ax=ax, x="n", y="time", color="device", marker="o")

    hue_name = "Device and Machine"
    df[hue_name] = df[["device", "machine"]].apply(
        lambda x: f"{x['device']}, {x['machine']}", axis=1
    )

    # grid = sns.FacetGrid(df, col="dtype", hue="hue", sharey=False)
    # grid.map(sns.lineplot, "n", "time", style="device",
    # markers={"cuda":"o", "cpu":"x"}, dashes=False)
    # grid.set(xscale="log", yscale="log")
    # grid.add_legend()
    hue_unique = df[hue_name].unique()
    df["device"].unique()
    df["machine"].unique()

    sns.set_theme()
    # sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    g = sns.relplot(
        data=df,
        x="n",
        y="time",
        hue=hue_name,
        style=hue_name,
        col="dtype",
        kind="line",
        markers={k: "o" if "cuda" in k else "X" for k in hue_unique},
        # dashes={k: [(0, ()), (0,(1,1)), (0, (5,5)),
        # (0, (3,5,1,5))][machine_unique.tolist().index(k[1])] for k in hue_unique},
        # palette=sns.color_palette("Greys"),
    )
    g.set_ylabels("Time (s)")
    g.set(xscale="log", yscale="log")

    # plt.tight_layout()
    [
        ax.xaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)
        for ax in g.axes.flatten()
    ]
    [
        ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)
        for ax in g.axes.flatten()
    ]
    plt.savefig("hpc/time_vs_n.svg")
    plt.savefig("hpc/time_vs_n.png")
    plt.savefig("hpc/time_vs_n.eps")


@app.command()
def benchmark() -> None:
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    LOG.info(f"{torch.get_num_threads()=}")
    ns = 2 ** np.linspace(5, 15, 20)
    ns = [int(n) for n in ns]
    # ks = 10 ** np.arange(1, 7, dtype=int)
    ks = [300]
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        devices = [
            torch.device("cuda"),
            torch.device("cpu"),
        ]
    else:
        devices = [torch.device("cpu")]
    dtypes = [torch.float32, torch.float64]
    for i in range(3):
        for dtype in dtypes:
            try:
                for n in ns:
                    for k in ks:
                        for device in devices:
                            with timer() as t:
                                # tracemalloc.reset_peak()
                                # tracemalloc.start()
                                uinf = get_uinf(
                                    k, n, (1, 0), (1, 0), 1, device=device, dtype=dtype
                                )
                                # tracemalloc.stop()
                            LOG.info(
                                f"n={n}, k={k}, "
                                f"device={device}, dtype={dtype}: {t.elapsed:.3f} s, "
                                f"uint={uinf.item()}, "
                                # f"max_memory={tracemalloc.get_traced_memory()[1]
                                # / 1024 ** 2:.3f} GB, "
                                f"max_memory_cached="
                                f"{torch.cuda.max_memory_cached() / 1024**3:.3f} GB, "
                                f"max_memory_allocated="
                                f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB, "
                            )
                            with Path(f"./result-{num_threads}.csv").open("a") as f:
                                f.write(
                                    f"{n},{k},{device},{dtype},{t.elapsed},{uinf.item()},"
                                    f"{torch.cuda.max_memory_cached()},"
                                    f"{torch.cuda.max_memory_allocated()}\n"
                                )
            except RuntimeError as e:
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_cached()
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.reset_peak_memory_stats()
                LOG.exception(e)


if __name__ == "__main__":
    app()
