"""
Visualization utilities for AutoConjecture Phase 4.

Provides:
  - Training curve plots (matplotlib / plotly)
  - Proof tree visualization (networkx / plotly)
  - Knowledge base analysis and statistics
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_kb(kb_path: str) -> Dict[str, Any]:
    """Load a knowledge-base JSON checkpoint."""
    with open(kb_path) as f:
        return json.load(f)


def load_stats_files(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    Scan *checkpoint_dir* for Phase-3 stats JSON files and return them
    sorted by (epoch, cycle).
    """
    pattern = os.path.join(checkpoint_dir, "*_stats_*.json")
    paths = sorted(glob.glob(pattern))
    records = []
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            data["_file"] = os.path.basename(p)
            records.append(data)
        except Exception:
            continue
    # Sort by epoch then cycle
    records.sort(key=lambda r: (r.get("epoch", 0), r.get("cycle", 0)))
    return records


def load_metrics_json(metrics_path: str) -> Dict[str, Any]:
    """Load a metrics JSON saved by MetricsTracker."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def find_latest_kb(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the most recently modified KB JSON in *checkpoint_dir*."""
    pattern = os.path.join(checkpoint_dir, "*_kb_*.json")
    paths = glob.glob(pattern)
    if not paths:
        # Phase-1 style: epoch_N_cycle_M.json
        pattern = os.path.join(checkpoint_dir, "epoch_*.json")
        paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def list_kb_files(checkpoint_dir: str) -> List[str]:
    """Return all KB JSON files in *checkpoint_dir*, newest first."""
    patterns = [
        os.path.join(checkpoint_dir, "*_kb_*.json"),
        os.path.join(checkpoint_dir, "epoch_*.json"),
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths = list(set(paths))
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths


# ---------------------------------------------------------------------------
# Training curve visualisations (Plotly – interactive)
# ---------------------------------------------------------------------------

def plot_kb_growth(stats: List[Dict], title: str = "Knowledge Base Growth") -> go.Figure:
    """Line chart of KB size over training steps from stats records."""
    if not stats:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    steps = list(range(len(stats)))
    kb_sizes = [r.get("kb_size", 0) for r in stats]
    labels = [f"e{r.get('epoch',0)} c{r.get('cycle',0)}" for r in stats]

    fig = go.Figure(go.Scatter(
        x=steps, y=kb_sizes, mode="lines+markers",
        hovertext=labels, hoverinfo="text+y",
        line=dict(color="#4C78A8", width=2),
        marker=dict(size=4),
        name="KB size",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Checkpoint #",
        yaxis_title="Theorems in KB",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_success_rates(
    stats: List[Dict],
    title: str = "Proof Success Rates",
) -> go.Figure:
    """Dual-line chart: RL success rate and heuristic success rate."""
    if not stats:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    steps = list(range(len(stats)))
    labels = [f"e{r.get('epoch',0)} c{r.get('cycle',0)}" for r in stats]

    fig = go.Figure()
    for key, color, name in [
        ("total_proved_rl", "#F58518", "RL prover"),
        ("total_proved_heuristic", "#54A24B", "Heuristic prover"),
        ("total_proved", "#4C78A8", "Total proved"),  # Phase-1/2
    ]:
        vals = [r.get(key) for r in stats]
        if any(v is not None for v in vals):
            fig.add_trace(go.Scatter(
                x=steps, y=[v or 0 for v in vals],
                mode="lines+markers",
                name=name,
                hovertext=labels,
                hoverinfo="text+y",
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Checkpoint #",
        yaxis_title="Cumulative proofs",
        legend=dict(orientation="h", y=-0.25),
        template="plotly_white",
        height=350,
    )
    return fig


def plot_ppo_metrics(stats: List[Dict], title: str = "PPO Training Metrics") -> go.Figure:
    """
    Subplot grid for PPO policy loss, value loss, entropy, and KL divergence.
    Only relevant for Phase-3 stats that include these keys.
    """
    ppo_keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "ppo_updates"]
    available = [k for k in ppo_keys if any(k in r for r in stats)]

    if not available:
        fig = go.Figure()
        fig.update_layout(title="No PPO metrics available")
        return fig

    # Show ppo_updates as a single line; the rest as a 2×2 grid
    plot_keys = [k for k in available if k != "ppo_updates"]
    if not plot_keys:
        plot_keys = ["ppo_updates"]

    n = len(plot_keys)
    cols = min(n, 2)
    rows = (n + 1) // 2
    steps = list(range(len(stats)))
    labels = [f"e{r.get('epoch',0)} c{r.get('cycle',0)}" for r in stats]

    subplot_titles = [k.replace("_", " ").title() for k in plot_keys]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    colors = ["#E45756", "#4C78A8", "#72B7B2", "#54A24B", "#F58518"]
    for i, key in enumerate(plot_keys):
        row, col = divmod(i, cols)
        vals = [r.get(key, None) for r in stats]
        if any(v is not None for v in vals):
            fig.add_trace(
                go.Scatter(
                    x=steps, y=[v or 0 for v in vals],
                    mode="lines",
                    name=key,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertext=labels,
                    hoverinfo="text+y",
                    showlegend=False,
                ),
                row=row + 1, col=col + 1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=200 * rows + 80,
    )
    return fig


def plot_complexity_distribution(theorems: List[Dict], title: str = "Complexity Distribution") -> go.Figure:
    """Histogram of theorem complexity values."""
    complexities = [t.get("complexity", 0) for t in theorems if "complexity" in t]
    if not complexities:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    fig = px.histogram(
        x=complexities, nbins=20,
        title=title,
        labels={"x": "Complexity", "y": "Count"},
        color_discrete_sequence=["#4C78A8"],
        template="plotly_white",
    )
    fig.update_layout(height=320)
    return fig


def plot_proof_length_distribution(theorems: List[Dict], title: str = "Proof Length Distribution") -> go.Figure:
    """Histogram of proof lengths."""
    lengths = [t.get("proof_length", 0) for t in theorems if "proof_length" in t]
    if not lengths:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    fig = px.histogram(
        x=lengths, nbins=15,
        title=title,
        labels={"x": "Proof Length (steps)", "y": "Count"},
        color_discrete_sequence=["#54A24B"],
        template="plotly_white",
    )
    fig.update_layout(height=320)
    return fig


def plot_discovery_timeline(theorems: List[Dict], title: str = "Theorem Discovery Timeline") -> go.Figure:
    """Scatter of theorem index vs epoch/cycle when it was discovered."""
    if not theorems:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    xs, ys, labels, complexities = [], [], [], []
    for i, t in enumerate(theorems):
        epoch = t.get("epoch", 0)
        cycle = t.get("cycle", 0)
        xs.append(epoch + cycle / 1000.0)  # fractional epoch for x-axis
        ys.append(i + 1)
        labels.append(t.get("statement", "")[:50])
        complexities.append(t.get("complexity", 5))

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            color=complexities,
            colorscale="Viridis",
            size=8,
            showscale=True,
            colorbar=dict(title="Complexity"),
        ),
        text=labels,
        hoverinfo="text+x+y",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch (fractional)",
        yaxis_title="Theorem #",
        template="plotly_white",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Proof tree visualisation
# ---------------------------------------------------------------------------

def parse_proof_steps(proof_steps: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parse proof-step strings of the form "  <tactic> → <goal>" or
    "  <tactic> → QED" into (tactic, goal_before, goal_after) tuples.

    Returns list of (node_id, label, parent_id) entries for tree building.
    """
    nodes: List[Tuple[str, str, str]] = []
    for i, step in enumerate(proof_steps):
        step = step.strip()
        # Split on " → " or " -> "
        for sep in (" → ", " -> "):
            if sep in step:
                tactic_part, goal_part = step.split(sep, 1)
                tactic = tactic_part.strip()
                goal = goal_part.strip()
                nodes.append((f"step_{i}", f"{tactic}\n{goal}", f"step_{i-1}" if i > 0 else "root"))
                break
        else:
            nodes.append((f"step_{i}", step[:60], f"step_{i-1}" if i > 0 else "root"))
    return nodes


def build_proof_tree(theorem: Dict) -> go.Figure:
    """
    Build an interactive Plotly tree diagram for a single theorem's proof.

    Args:
        theorem: dict with keys 'statement', 'proof_steps', 'proof_length', etc.
    """
    statement = theorem.get("statement", "?")
    proof_steps = theorem.get("proof_steps", [])

    if not proof_steps:
        fig = go.Figure()
        fig.add_annotation(
            text="No proof steps available for this theorem.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title=f"Proof of: {statement[:80]}", height=300)
        return fig

    # Build directed graph (root → step_0 → step_1 → … → QED)
    G = nx.DiGraph()
    root_id = "root"
    root_label = f"Goal\n{statement[:60]}"
    G.add_node(root_id, label=root_label, depth=0)

    prev_id = root_id
    for i, step_str in enumerate(proof_steps):
        step_str = step_str.strip()
        node_id = f"s{i}"
        is_qed = "QED" in step_str or "qed" in step_str.lower()

        # Derive label: tactic + resulting state
        for sep in (" → ", " -> "):
            if sep in step_str:
                tactic, rest = step_str.split(sep, 1)
                label = f"[{tactic.strip()}]\n{rest.strip()[:50]}"
                break
        else:
            label = step_str[:60]

        if is_qed:
            label = f"[{step_str.split(chr(8594))[0].strip() if '→' in step_str else step_str[:20]}]\n✓ QED"

        G.add_node(node_id, label=label, depth=i + 1, is_qed=is_qed)
        G.add_edge(prev_id, node_id)
        prev_id = node_id

    # Compute positions using a simple layered layout
    pos: Dict[str, Tuple[float, float]] = {}
    nodes_by_depth: Dict[int, List[str]] = {}
    for node, data in G.nodes(data=True):
        d = data.get("depth", 0)
        nodes_by_depth.setdefault(d, []).append(node)

    max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0
    for depth, nodes_at_depth in nodes_by_depth.items():
        y = -depth  # grow downward
        n = len(nodes_at_depth)
        for j, node in enumerate(nodes_at_depth):
            x = j - (n - 1) / 2.0
            pos[node] = (x, y)

    # Build Plotly scatter + line traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#AAAAAA"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [G.nodes[n].get("label", n) for n in G.nodes()]
    is_qed_flags = [G.nodes[n].get("is_qed", False) for n in G.nodes()]
    node_colors = [
        "#54A24B" if qed else ("#4C78A8" if n == root_id else "#F58518")
        for n, qed in zip(G.nodes(), is_qed_flags)
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=18, color=node_colors, line=dict(width=1, color="white")),
        text=[lbl.split("\n")[0] for lbl in node_labels],
        textposition="top center",
        hovertext=node_labels,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Proof tree: {statement[:70]}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        height=max(400, 120 * (max_depth + 2)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib static exports (for non-Streamlit use)
# ---------------------------------------------------------------------------

def save_training_curves(
    stats: List[Dict],
    output_path: str = "data/logs/training_curves.png",
) -> str:
    """
    Save a static matplotlib figure with training curves to *output_path*.
    Returns the path.
    """
    if not stats:
        return ""

    steps = list(range(len(stats)))
    kb_sizes = [r.get("kb_size", 0) for r in stats]
    rl_proofs = [r.get("total_proved_rl", 0) for r in stats]
    h_proofs = [r.get("total_proved_heuristic", 0) for r in stats]
    total_proved = [r.get("total_proved", 0) or (rl + h)
                    for r, rl, h in zip(stats, rl_proofs, h_proofs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("AutoConjecture Training Progress", fontsize=13, fontweight="bold")

    axes[0].plot(steps, kb_sizes, color="#4C78A8", linewidth=2, marker="o", markersize=3)
    axes[0].set_title("Knowledge Base Growth")
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("Theorems")
    axes[0].grid(True, alpha=0.3)

    if any(rl_proofs) or any(h_proofs):
        axes[1].plot(steps, rl_proofs, label="RL prover", color="#F58518", linewidth=2)
        axes[1].plot(steps, h_proofs, label="Heuristic", color="#54A24B", linewidth=2)
        axes[1].legend()
    else:
        axes[1].plot(steps, total_proved, color="#4C78A8", linewidth=2)

    axes[1].set_title("Cumulative Proofs")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("Proofs")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_kb_analysis(
    kb_data: Dict,
    output_path: str = "data/logs/kb_analysis.png",
) -> str:
    """
    Save a static matplotlib figure summarising a KB checkpoint.
    Returns the path.
    """
    theorems = kb_data.get("theorems", [])
    if not theorems:
        return ""

    complexities = [t.get("complexity", 0) for t in theorems]
    lengths = [t.get("proof_length", 0) for t in theorems]
    epochs = [t.get("epoch", 0) for t in theorems]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Knowledge Base Analysis  ({len(theorems)} theorems)",
        fontsize=13, fontweight="bold",
    )

    axes[0].hist(complexities, bins=20, color="#4C78A8", edgecolor="white")
    axes[0].set_title("Complexity Distribution")
    axes[0].set_xlabel("Complexity")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(lengths, bins=15, color="#54A24B", edgecolor="white")
    axes[1].set_title("Proof Length Distribution")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    if len(set(epochs)) > 1:
        epoch_counts: Dict[int, int] = {}
        for e in epochs:
            epoch_counts[e] = epoch_counts.get(e, 0) + 1
        sorted_epochs = sorted(epoch_counts.keys())
        axes[2].bar(sorted_epochs, [epoch_counts[e] for e in sorted_epochs],
                    color="#F58518", edgecolor="white")
        axes[2].set_title("Theorems Discovered per Epoch")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Count")
        axes[2].grid(True, alpha=0.3, axis="y")
    else:
        axes[2].text(0.5, 0.5, "Single epoch\ndata", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title("Per-Epoch Breakdown")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
