from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ("depth", "number_of_nodes", "max_breadth")
BOOT_METRICS = ("depth", "max_breadth")
VERSION = "4-clean-layout-horizontal-bootstrap"


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _filename_meta(path):
    result = {}
    for part in path.stem.split("__"):
        for key in ("genre", "source", "n_sentences"):
            if part.startswith(key + "-"):
                result[key] = part[len(key) + 1:]
    return result


def _tree_metrics(root):
    queue = deque([(root, 0)])
    levels = Counter()

    while queue:
        node, depth = queue.popleft()
        levels[depth] += 1
        queue.extend((child, depth + 1) for child in node)

    return {
        "depth": max(levels),
        "number_of_nodes": sum(levels.values()),
        "max_breadth": max(levels.values()),
    }


def load_tree_metrics(
    forests_dir, *, genre=None, n_sentences=None, sources=None
):
    forests_dir = Path(forests_dir)
    if not forests_dir.is_dir():
        raise FileNotFoundError(forests_dir)

    allowed = set(sources) if sources is not None else None
    rows = []
    matched_files = []

    for xml in sorted(forests_dir.glob("*.xml")):
        meta = _filename_meta(xml)

        if not all(
            k in meta for k in ("genre", "source", "n_sentences")
        ):
            raise ValueError(
                f"Missing metadata in forest filename: {xml.name}"
            )

        if genre is not None and meta["genre"] != str(genre):
            continue
        if (
            n_sentences is not None
            and meta["n_sentences"] != str(n_sentences)
        ):
            continue
        if allowed is not None and meta["source"] not in allowed:
            continue

        matched_files.append(xml)
        stack = []
        found = 0

        for event, elem in ET.iterparse(
            xml, events=("start", "end")
        ):
            if event == "start":
                stack.append(elem)
                continue

            if (
                elem.tag.startswith("DOC_")
                and len(stack) == 2
                and stack[0].tag == "forest"
            ):
                if len(elem) != 1:
                    raise ValueError(
                        f"Expected one RST root: {xml.name}/{elem.tag}"
                    )

                rows.append({
                    "genre": meta["genre"],
                    "source": meta["source"],
                    "n_sentences": int(meta["n_sentences"]),
                    "forest_file": xml.name,
                    "tree_id": elem.tag,
                    **_tree_metrics(elem[0]),
                })

                found += 1
                stack[0].remove(elem)
                elem.clear()

            stack.pop()

        if found == 0:
            raise ValueError(f"No DOC trees in {xml}")

    if not matched_files:
        raise ValueError(
            f"No XML forests match genre={genre}, "
            f"n_sentences={n_sentences}, sources={sources}"
        )

    data = pd.DataFrame(rows)

    if allowed is not None and set(data.source) != allowed:
        raise ValueError(
            f"Missing sources: {sorted(allowed - set(data.source))}"
        )

    if data.duplicated(["forest_file", "tree_id"]).any():
        raise ValueError("Duplicate tree identifiers")

    return data


def _ci(draws, confidence):
    alpha = (1.0 - confidence) / 2.0
    return np.quantile(
        draws, [alpha, 1.0 - alpha]
    ).tolist()


def _bootstrap_mean(values, rng, n_boot):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        raise ValueError("At least two trees are required")

    result = np.empty(n_boot)

    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        indices = rng.integers(
            0, n, size=(stop - start, n)
        )
        result[start:stop] = values[indices].mean(axis=1)

    return result


def calculate_statistics(
    data, *, n_boot=10000, confidence=0.95,
    seed=42, human_source="Human",
    delta_method="independent"
):
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level")
    if delta_method not in ("independent", "paired"):
        raise ValueError("Invalid delta_method")

    groups = {
        source: group
        for source, group in data.groupby("source", sort=True)
    }

    if human_source not in groups:
        raise ValueError(
            f"Human reference '{human_source}' not found"
        )

    rng = np.random.default_rng(seed)

    summaries = []
    means = []
    deltas = []
    mean_draws = []
    delta_draws = []
    boot = {}

    for source, group in groups.items():
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            n = len(values)

            summaries.append({
                "source": source,
                "metric": metric,
                "n": n,
                "mean": values.mean(),
                "sd": values.std(ddof=1) if n > 1 else np.nan,
                "sem": (
                    values.std(ddof=1) / np.sqrt(n)
                    if n > 1 else np.nan
                ),
                "median": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
            })

            if metric not in BOOT_METRICS:
                continue

            draws = _bootstrap_mean(values, rng, n_boot)
            boot[(source, metric)] = draws

            low, high = _ci(draws, confidence)

            means.append({
                "source": source,
                "metric": metric,
                "n": n,
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "confidence": confidence,
            })

            mean_draws.extend(
                (source, metric, i, float(value))
                for i, value in enumerate(draws)
            )

    for source, group in groups.items():
        if source == human_source:
            continue

        for metric in BOOT_METRICS:
            human = groups[human_source]

            if delta_method == "paired":
                left = group.set_index("tree_id")
                right = human.set_index("tree_id")
                common = left.index.intersection(right.index)

                if len(common) < 2:
                    raise ValueError(
                        f"Cannot pair {source} with {human_source}"
                    )

                differences = (
                    left.loc[common, metric].to_numpy(float)
                    - right.loc[common, metric].to_numpy(float)
                )

                draws = _bootstrap_mean(
                    differences, rng, n_boot
                )
                estimate = differences.mean()
                n_source = n_reference = len(differences)

            else:
                draws = (
                    boot[(source, metric)]
                    - boot[(human_source, metric)]
                )

                estimate = (
                    group[metric].mean()
                    - human[metric].mean()
                )

                n_source = len(group)
                n_reference = len(human)

            low, high = _ci(draws, confidence)

            deltas.append({
                "source": source,
                "reference": human_source,
                "metric": metric,
                "n_source": n_source,
                "n_reference": n_reference,
                "delta": estimate,
                "ci_low": low,
                "ci_high": high,
                "excludes_zero": bool(low > 0 or high < 0),
                "confidence": confidence,
                "bootstrap_method": delta_method,
            })

            delta_draws.extend(
                (source, human_source, metric, i, float(value))
                for i, value in enumerate(draws)
            )

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(means),
        pd.DataFrame(deltas),
        pd.DataFrame(
            mean_draws,
            columns=[
                "source", "metric", "iteration",
                "bootstrap_mean"
            ],
        ),
        pd.DataFrame(
            delta_draws,
            columns=[
                "source", "reference", "metric",
                "iteration", "bootstrap_delta"
            ],
        ),
    )


def _title(genre, n_sentences, data):
    counts = data.groupby("source").size()

    if counts.nunique() == 1:
        sample_text = (
            f"{int(counts.iloc[0])} samples per source"
        )
    else:
        sample_text = (
            f"{len(data)} samples across "
            f"{len(counts)} sources"
        )

    genre_name = str(genre).replace("_", " ").title()

    if genre_name.lower() == "reddit":
        genre_name += " posts"
    elif genre_name.lower() == "news":
        genre_name += " articles"
    elif genre_name.lower() == "abstracts":
        genre_name += " abstracts"

    return (
        f"Structural Analysis of {genre_name} "
        f"with {n_sentences} sentences "
        f"and {sample_text}"
    )


def _save(fig, folder, name, show):
    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        folder / f"{name}.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        folder / f"{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)


def _horizontal_ci(ax, estimate, low, high, y, color):
    ax.errorbar(
        estimate, y,
        xerr=[
            [max(0, estimate - low)],
            [max(0, high - estimate)]
        ],
        fmt="o",
        color=color,
        markersize=6,
        capsize=4,
        elinewidth=1.8,
        zorder=3,
    )


def plot_results(
    data, summary, means, deltas, folder, *,
    show=True, hex_gridsize=14, context="",
    confidence=0.95, n_boot=10000,
    delta_method="independent",
    human_source="Human"
):
    from matplotlib.colors import Normalize

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    sources = sorted(data.source.unique())

    if human_source in sources:
        sources.remove(human_source)
        sources.insert(0, human_source)

    cmap = plt.get_cmap("tab10")
    colors = {
        source: cmap(i % 10)
        for i, source in enumerate(sources)
    }

    labels = {
        "depth": "Tree depth",
        "number_of_nodes": "Number of nodes",
        "max_breadth": "Maximum breadth",
    }

    alpha = 1.0 - confidence
    boot_text = (
        f"Percentile bootstrap, B={n_boot:,}, "
        f"{confidence:.0%} CI, α={alpha:.2f}"
    )

    # --------------------------------------------------------
    # 01 — Structural boxplots
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5),
        layout="constrained"
    )

    for ax, metric in zip(axes, METRICS):
        values = [
            data.loc[data.source == src, metric].to_numpy()
            for src in sources
        ]

        bp = ax.boxplot(
            values,
            tick_labels=sources,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black"},
        )

        for patch, src in zip(bp["boxes"], sources):
            patch.set_facecolor(colors[src])
            patch.set_alpha(0.55)

        ax.set_ylabel(labels[metric])
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(context, fontsize=13, fontweight="semibold")

    _save(
        fig, folder,
        "01_structural_boxplots", show
    )

    # --------------------------------------------------------
    # 02 — Hexbin depth vs maximum breadth
    # --------------------------------------------------------

    xlim = (
        data.depth.min() - 0.5,
        data.depth.max() + 0.5
    )
    ylim = (
        data.max_breadth.min() - 0.5,
        data.max_breadth.max() + 0.5
    )

    extent = (*xlim, *ylim)

    # Determine a shared color scale
    maximum_count = 1

    for src in sources:
        subset = data[data.source == src]

        temp_fig, temp_ax = plt.subplots()
        hb = temp_ax.hexbin(
            subset.depth,
            subset.max_breadth,
            gridsize=hex_gridsize,
            extent=extent,
            mincnt=1,
        )

        counts = hb.get_array()
        if len(counts):
            maximum_count = max(
                maximum_count, int(counts.max())
            )

        plt.close(temp_fig)

    fig, axes = plt.subplots(
        1, len(sources),
        figsize=(4 * len(sources), 4.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    axes = np.atleast_1d(axes)
    norm = Normalize(
        vmin=1,
        vmax=max(2, maximum_count)
    )

    for ax, src in zip(axes, sources):
        subset = data[data.source == src]

        hb = ax.hexbin(
            subset.depth,
            subset.max_breadth,
            gridsize=hex_gridsize,
            extent=extent,
            mincnt=1,
            cmap="viridis",
            norm=norm,
        )

        ax.set_title(src, fontsize=11)
        ax.set_xlabel("Tree depth")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    axes[0].set_ylabel("Maximum breadth")

    fig.colorbar(
        hb, ax=axes.tolist(),
        label="Trees per hexagon",
        shrink=0.8,
    )

    fig.suptitle(
        context, fontsize=13, fontweight="semibold"
    )

    _save(
        fig, folder,
        "02_depth_breadth_hexbin", show
    )

    # --------------------------------------------------------
    # 03 — Structural centroids, mean ± SEM
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6),
        layout="constrained"
    )

    for src in sources:
        subset = data[data.source == src]
        n = len(subset)

        if n < 2:
            raise ValueError(
                f"SEM requires at least two trees: {src}"
            )

        mean_depth = subset.depth.mean()
        mean_breadth = subset.max_breadth.mean()

        sem_depth = (
            subset.depth.std(ddof=1) / np.sqrt(n)
        )
        sem_breadth = (
            subset.max_breadth.std(ddof=1) / np.sqrt(n)
        )

        ax.errorbar(
            mean_depth,
            mean_breadth,
            xerr=sem_depth,
            yerr=sem_breadth,
            fmt="o",
            color=colors[src],
            markersize=7,
            capsize=4,
            label=src,
        )

    ax.set_xlabel("Mean tree depth (± SEM)")
    ax.set_ylabel("Mean maximum breadth (± SEM)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle(
        context, fontsize=13, fontweight="semibold"
    )

    # Retain the previous filename for overwrite compatibility
    _save(
        fig, folder,
        "03_centroids_sd", show
    )

    # --------------------------------------------------------
    # 04 — Bootstrap means: horizontal confidence intervals
    # --------------------------------------------------------

    mean_lookup = means.set_index(
        ["source", "metric"]
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        layout="constrained"
    )

    for ax, metric in zip(axes, BOOT_METRICS):
        human = mean_lookup.loc[
            (human_source, metric)
        ]

        # Human CI boundaries
        for boundary in (
            human["ci_low"],
            human["ci_high"]
        ):
            ax.axvline(
                boundary,
                color=colors[human_source],
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                zorder=1,
            )

        for j, src in enumerate(sources):
            row = mean_lookup.loc[(src, metric)]

            _horizontal_ci(
                ax,
                row["mean"],
                row["ci_low"],
                row["ci_high"],
                j,
                colors[src],
            )

        ax.set_yticks(range(len(sources)), sources)
        ax.invert_yaxis()
        ax.set_xlabel(
            f"Mean {labels[metric].lower()}"
        )
        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.12, y=0.12)

    fig.suptitle(
        f"{context} — {boot_text}",
        fontsize=12,
        fontweight="semibold",
    )

    _save(
        fig, folder,
        "04_bootstrap_means", show
    )

    # --------------------------------------------------------
    # 05 — Bootstrap deltas: LLM minus Human
    # --------------------------------------------------------

    other_sources = [
        src for src in sources
        if src != human_source
    ]

    delta_lookup = deltas.set_index(
        ["source", "metric"]
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        layout="constrained"
    )

    for ax, metric in zip(axes, BOOT_METRICS):
        ax.axvline(
            0,
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            zorder=1,
        )

        for j, src in enumerate(other_sources):
            row = delta_lookup.loc[(src, metric)]

            _horizontal_ci(
                ax,
                row["delta"],
                row["ci_low"],
                row["ci_high"],
                j,
                colors[src],
            )

        ax.set_yticks(
            range(len(other_sources)),
            other_sources
        )
        ax.invert_yaxis()

        ax.set_xlabel(
            f"Mean {labels[metric].lower()} "
            f"difference (LLM − {human_source})"
        )

        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.12, y=0.15)

    fig.suptitle(
        f"{context} — {boot_text}, "
        f"{delta_method} resampling",
        fontsize=12,
        fontweight="semibold",
    )

    _save(
        fig, folder,
        "05_bootstrap_deltas", show
    )


def _sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_rst_trees(
    *,
    forests_dir,
    results_dir,
    genre,
    n_sentences,
    sources=None,
    dataset_name=None,
    n_boot=10000,
    confidence=0.95,
    seed=42,
    human_source="Human",
    delta_method="independent",
    show=True,
    hex_gridsize=14,
    rebuild=False,
    prompting=None,
    attack=None,
    decoding=None,
    repetition_penalty=None,
    matched=None,
    sampling=None,
    tree_set=None,
):
    forests_dir = Path(forests_dir)

    files = sorted(forests_dir.glob("*.xml"))
    if not files:
        raise FileNotFoundError(
            f"No XML forests in {forests_dir}"
        )

    selected_files = []

    for path in files:
        meta = _filename_meta(path)

        if genre is not None and meta.get("genre") != str(genre):
            continue

        if (
            n_sentences is not None
            and meta.get("n_sentences") != str(n_sentences)
        ):
            continue

        if (
            sources is not None
            and meta.get("source") not in set(sources)
        ):
            continue

        selected_files.append({
            "filename": path.name,
            "sha256": _sha256(path),
        })

    if not selected_files:
        raise ValueError(
            "No XML forests match the requested filters"
        )

    if dataset_name is None:
        dataset_name = forests_dir.parent.name

    out = (
        Path(results_dir)
        / "structural_analysis"
        / _safe(dataset_name)
    )

    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "analysis_version": VERSION,
        "genre": genre,
        "n_sentences": n_sentences,
        "sources": (
            sorted(sources) if sources is not None else None
        ),
        "n_boot": n_boot,
        "confidence": confidence,
        "seed": seed,
        "human_source": human_source,
        "delta_method": delta_method,
        "hex_gridsize": hex_gridsize,
        "prompting": prompting,
        "attack": attack,
        "decoding": decoding,
        "repetition_penalty": repetition_penalty,
        "matched": matched,
        "sampling": sampling,
        "tree_set": tree_set,
        "forest_inputs": selected_files,
        "metric_definition": (
            "Root depth 0; count rhetorical nodes and EDU leaves; "
            "exclude DOC and forest wrappers"
        ),
    }

    fingerprint = hashlib.sha256(
        json.dumps(
            settings,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    manifest_path = out / "analysis_manifest.json"

    table_names = (
        "tree_metrics",
        "source_summary",
        "bootstrap_means",
        "bootstrap_deltas",
        "bootstrap_mean_draws",
        "bootstrap_delta_draws",
    )

    figure_names = (
        "01_structural_boxplots",
        "02_depth_breadth_hexbin",
        "03_centroids_sd",
        "04_bootstrap_means",
        "05_bootstrap_deltas",
    )

    ready = (
        manifest_path.is_file()
        and all(
            (out / f"{name}.csv").is_file()
            for name in table_names
        )
        and all(
            (out / "figures" / f"{name}.{ext}").is_file()
            for name in figure_names
            for ext in ("pdf", "png")
        )
    )

    if ready and not rebuild:
        previous = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if previous.get("fingerprint") == fingerprint:
            print(
                f"Structural analysis: loaded cache from {out}"
            )

            tables = {
                name: pd.read_csv(out / f"{name}.csv")
                for name in table_names
            }

            if show:
                from IPython.display import display, Image

                for name in figure_names:
                    display(
                        Image(
                            filename=str(
                                out / "figures" / f"{name}.png"
                            )
                        )
                    )

            return out, {
                "trees": tables["tree_metrics"],
                "summary": tables["source_summary"],
                "bootstrap_means": tables["bootstrap_means"],
                "bootstrap_deltas": tables["bootstrap_deltas"],
            }

        print(
            "Inputs or settings changed: rebuilding analysis"
        )

    # Recompute from individual RST trees
    data = load_tree_metrics(
        forests_dir,
        genre=genre,
        n_sentences=n_sentences,
        sources=sources,
    )

    if data.source.nunique() < 2:
        raise ValueError(
            "At least two sources are required"
        )

    (
        summary,
        means,
        deltas,
        mean_draws,
        delta_draws,
    ) = calculate_statistics(
        data,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
        human_source=human_source,
        delta_method=delta_method,
    )

    tables = {
        "tree_metrics": data,
        "source_summary": summary,
        "bootstrap_means": means,
        "bootstrap_deltas": deltas,
        "bootstrap_mean_draws": mean_draws,
        "bootstrap_delta_draws": delta_draws,
    }

    # Overwrite existing CSV files
    for name, frame in tables.items():
        frame.to_csv(
            out / f"{name}.csv",
            index=False
        )

    flat_settings = {
        k: v
        for k, v in settings.items()
        if k != "forest_inputs"
    }

    flat_settings["sources"] = ",".join(
        sorted(data.source.unique())
    )
    flat_settings["n_trees"] = len(data)

    pd.DataFrame([flat_settings]).to_csv(
        out / "analysis_settings.csv",
        index=False,
    )

    context = _title(
        genre, n_sentences, data
    )

    plot_results(
        data,
        summary,
        means,
        deltas,
        out / "figures",
        show=show,
        hex_gridsize=hex_gridsize,
        context=context,
        confidence=confidence,
        n_boot=n_boot,
        delta_method=delta_method,
        human_source=human_source,
    )

    # Update manifest only after successful generation
    temporary = manifest_path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "settings": settings,
                "n_trees": len(data),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(manifest_path)

    print(
        f"Structural analysis: {len(data)} trees"
    )
    print(f"Results saved to: {out}")

    return out, {
        "trees": data,
        "summary": summary,
        "bootstrap_means": means,
        "bootstrap_deltas": deltas,
    }
