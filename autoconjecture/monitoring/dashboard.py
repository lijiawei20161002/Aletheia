"""
AutoConjecture Phase 4: Interactive Streamlit Dashboard.

Launch with:
    streamlit run scripts/dashboard.py

Or directly (from project root):
    python -m streamlit run src/monitoring/dashboard.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Allow importing from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.monitoring.visualizer import (
    build_proof_tree,
    find_latest_kb,
    list_kb_files,
    load_kb,
    load_metrics_json,
    load_stats_files,
    plot_complexity_distribution,
    plot_discovery_timeline,
    plot_kb_growth,
    plot_ppo_metrics,
    plot_proof_length_distribution,
    plot_success_rates,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoConjecture Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar – data source selection
# ---------------------------------------------------------------------------
def _sidebar() -> Dict[str, Any]:
    """Render sidebar and return selected data-source paths."""
    st.sidebar.title("⚙️  Data Sources")

    default_ckpt = os.path.join(_ROOT, "data", "checkpoints")
    default_logs = os.path.join(_ROOT, "data", "logs")

    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint directory",
        value=default_ckpt,
        help="Directory that contains *_kb_*.json, *_stats_*.json, etc.",
    )
    log_dir = st.sidebar.text_input(
        "Log directory",
        value=default_logs,
        help="Directory that contains metrics.json and *.log files.",
    )

    st.sidebar.divider()

    # Auto-refresh
    auto_refresh = st.sidebar.toggle("Auto-refresh (30 s)", value=False)
    if auto_refresh:
        import time
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("AutoConjecture  •  Phase 4 Monitoring")

    return {"checkpoint_dir": checkpoint_dir, "log_dir": log_dir}


# ---------------------------------------------------------------------------
# Tab helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _load_stats(checkpoint_dir: str) -> List[Dict]:
    return load_stats_files(checkpoint_dir)


@st.cache_data(ttl=30)
def _load_kb_cached(path: str) -> Dict:
    return load_kb(path)


@st.cache_data(ttl=30)
def _load_metrics(log_dir: str) -> Dict:
    return load_metrics_json(os.path.join(log_dir, "metrics.json"))


def _kb_selector(checkpoint_dir: str, sidebar_key: str = "kb_file") -> Optional[Dict]:
    """Render a KB file selector and return the loaded KB dict (or None)."""
    kb_files = list_kb_files(checkpoint_dir)
    if not kb_files:
        st.warning(f"No KB checkpoint files found in `{checkpoint_dir}`.")
        return None

    # Show only basenames in the selector, keep full path as map
    basenames = [os.path.basename(p) for p in kb_files]
    selected = st.selectbox(
        "Select KB checkpoint",
        basenames,
        index=0,
        key=sidebar_key,
        help="Newest files are listed first.",
    )
    full_path = kb_files[basenames.index(selected)]
    return _load_kb_cached(full_path)


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

def _tab_overview(checkpoint_dir: str, log_dir: str):
    st.header("📊  Overview")

    stats = _load_stats(checkpoint_dir)
    latest_kb_path = find_latest_kb(checkpoint_dir)
    kb_data = _load_kb_cached(latest_kb_path) if latest_kb_path else {}
    theorems = kb_data.get("theorems", [])

    # ── Top metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_theorems = len(theorems)
    with c1:
        st.metric("Theorems Proved", total_theorems)
    with c2:
        total_attempted = stats[-1].get("total_attempted", 0) if stats else 0
        total_proved_all = (
            (stats[-1].get("total_proved_rl", 0) or 0)
            + (stats[-1].get("total_proved_heuristic", 0) or 0)
            + (stats[-1].get("total_proved", 0) or 0)
        ) if stats else 0
        rate = total_proved_all / max(total_attempted, 1)
        st.metric("Overall Success Rate", f"{rate:.1%}")
    with c3:
        avg_cpx = (
            sum(t.get("complexity", 0) for t in theorems) / len(theorems)
            if theorems else 0.0
        )
        st.metric("Avg Complexity", f"{avg_cpx:.2f}")
    with c4:
        avg_len = (
            sum(t.get("proof_length", 0) for t in theorems) / len(theorems)
            if theorems else 0.0
        )
        st.metric("Avg Proof Length", f"{avg_len:.1f} steps")

    st.divider()

    # ── KB growth sparkline ──────────────────────────────────────────────────
    if stats:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_kb_growth(stats), use_container_width=True, key="ov_kb_growth")
        with col_b:
            st.plotly_chart(plot_success_rates(stats), use_container_width=True, key="ov_success_rates")
    else:
        st.info("No training stats found. Run a training script first, then "
                "refresh the dashboard.")

    # ── Recent theorems ──────────────────────────────────────────────────────
    if theorems:
        st.subheader("🆕  Most Recently Proved Theorems")
        recent = theorems[-10:][::-1]
        for t in recent:
            with st.expander(t.get("statement", "?")[:100]):
                col1, col2, col3 = st.columns(3)
                col1.metric("Complexity", f"{t.get('complexity', 0):.2f}")
                col2.metric("Proof length", t.get("proof_length", 0))
                col3.metric("Epoch / Cycle", f"{t.get('epoch', '?')} / {t.get('cycle', '?')}")
                steps = t.get("proof_steps", [])
                if steps:
                    st.code("\n".join(steps), language="text")


# ---------------------------------------------------------------------------
# Tab 2: Training Curves
# ---------------------------------------------------------------------------

def _tab_training_curves(checkpoint_dir: str, log_dir: str):
    st.header("📈  Training Curves")

    stats = _load_stats(checkpoint_dir)
    metrics = _load_metrics(log_dir)

    if not stats and not metrics:
        st.info("No stats or metrics data found. Run a training script first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_kb_growth(stats), use_container_width=True, key="tc_kb_growth")
    with col2:
        st.plotly_chart(plot_success_rates(stats), use_container_width=True, key="tc_success_rates")

    # ── PPO metrics (Phase 3 only) ───────────────────────────────────────────
    ppo_fig = plot_ppo_metrics(stats)
    st.plotly_chart(ppo_fig, use_container_width=True, key="tc_ppo_metrics")

    # ── Metrics JSON (from MetricsTracker) ───────────────────────────────────
    if metrics:
        st.subheader("📋  Logged Metrics")
        metric_names = list(metrics.get("metrics", {}).keys())
        if metric_names:
            selected = st.multiselect(
                "Select metrics to plot",
                metric_names,
                default=metric_names[:4],
            )
            steps = metrics.get("steps", [])
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = [
                "#4C78A8", "#F58518", "#54A24B", "#E45756",
                "#72B7B2", "#B279A2", "#FF9DA7", "#9D755D",
            ]
            for i, name in enumerate(selected):
                vals = metrics["metrics"][name]
                xs = steps[: len(vals)] if steps else list(range(len(vals)))
                fig.add_trace(go.Scatter(
                    x=xs, y=vals, mode="lines", name=name,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            fig.update_layout(
                title="Custom Metrics",
                template="plotly_white",
                height=380,
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig, use_container_width=True, key="tc_custom_metrics")

    # ── Raw stats table ──────────────────────────────────────────────────────
    if stats:
        with st.expander("Raw Stats Table"):
            df = pd.DataFrame(stats)
            # Drop internal metadata column
            df = df.drop(columns=["_file"], errors="ignore")
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Proof Explorer
# ---------------------------------------------------------------------------

def _tab_proof_explorer(checkpoint_dir: str):
    st.header("🌳  Proof Explorer")

    kb_data = _kb_selector(checkpoint_dir, sidebar_key="proof_kb")
    if kb_data is None:
        return

    theorems = kb_data.get("theorems", [])
    if not theorems:
        st.warning("The selected KB checkpoint contains no theorems.")
        return

    # Search / filter
    search = st.text_input("🔍  Filter theorems (substring match)", "")
    filtered = [
        t for t in theorems
        if search.lower() in t.get("statement", "").lower()
    ] if search else theorems

    if not filtered:
        st.info("No theorems match the filter.")
        return

    # Select one theorem
    options = {
        f"#{i+1}  {t.get('statement','')[:80]}": t
        for i, t in enumerate(filtered)
    }
    selection_label = st.selectbox("Select a theorem", list(options.keys()))
    selected_theorem = options[selection_label]

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Complexity", f"{selected_theorem.get('complexity', 0):.2f}")
    col2.metric("Proof length", selected_theorem.get("proof_length", 0))
    col3.metric("Epoch", selected_theorem.get("epoch", "?"))
    col4.metric("Cycle", selected_theorem.get("cycle", "?"))

    # Proof tree
    st.plotly_chart(build_proof_tree(selected_theorem), use_container_width=True, key="pe_proof_tree")

    # Raw proof steps
    steps = selected_theorem.get("proof_steps", [])
    if steps:
        with st.expander("Proof Steps (text)"):
            for i, s in enumerate(steps):
                st.text(f"  {i+1}. {s}")
    else:
        st.caption("No proof steps stored for this theorem.")


# ---------------------------------------------------------------------------
# Tab 4: Knowledge Base Browser
# ---------------------------------------------------------------------------

def _tab_kb_browser(checkpoint_dir: str):
    st.header("📚  Knowledge Base Browser")

    kb_data = _kb_selector(checkpoint_dir, sidebar_key="kb_browser_kb")
    if kb_data is None:
        return

    theorems = kb_data.get("theorems", [])
    axioms = kb_data.get("axioms", [])
    metadata = kb_data.get("metadata", {})

    if not theorems:
        st.warning("The selected KB checkpoint contains no theorems.")
        return

    # ── Summary cards ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Theorems", len(theorems))
    c2.metric("Axioms", len(axioms))
    c3.metric("Saved At", metadata.get("saved_at", "unknown")[:19])

    st.divider()

    # ── Distribution plots ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_complexity_distribution(theorems), use_container_width=True, key="kb_complexity_dist")
    with col_b:
        st.plotly_chart(plot_proof_length_distribution(theorems), use_container_width=True, key="kb_proof_length_dist")

    st.plotly_chart(plot_discovery_timeline(theorems), use_container_width=True, key="kb_discovery_timeline")

    st.divider()

    # ── Filterable, sortable table ───────────────────────────────────────────
    st.subheader("Theorem Table")

    # Complexity range filter
    all_cpx = [t.get("complexity", 0) for t in theorems]
    min_cpx, max_cpx = (min(all_cpx), max(all_cpx)) if all_cpx else (0.0, 10.0)
    cpx_range = st.slider(
        "Complexity range",
        float(min_cpx), float(max_cpx),
        (float(min_cpx), float(max_cpx)),
        step=0.5,
    )
    search_kb = st.text_input("🔍  Search statements", "")

    rows = []
    for i, t in enumerate(theorems):
        cpx = t.get("complexity", 0)
        if not (cpx_range[0] <= cpx <= cpx_range[1]):
            continue
        stmt = t.get("statement", "")
        if search_kb and search_kb.lower() not in stmt.lower():
            continue
        rows.append({
            "#": i + 1,
            "Statement": stmt,
            "Complexity": round(cpx, 2),
            "Proof Length": t.get("proof_length", 0),
            "Epoch": t.get("epoch", 0),
            "Cycle": t.get("cycle", 0),
            "Discovered": t.get("timestamp", "")[:19],
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=420)
        st.caption(f"Showing {len(rows)} / {len(theorems)} theorems")
    else:
        st.info("No theorems match the selected filters.")

    # ── Axioms ───────────────────────────────────────────────────────────────
    if axioms:
        with st.expander(f"Axioms ({len(axioms)})"):
            for ax in axioms:
                st.code(ax, language="text")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    st.title("🔬  AutoConjecture  —  Phase 4 Dashboard")
    st.caption("Real-time monitoring of mathematical theorem discovery")

    cfg = _sidebar()
    checkpoint_dir = cfg["checkpoint_dir"]
    log_dir = cfg["log_dir"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Overview",
        "📈  Training Curves",
        "🌳  Proof Explorer",
        "📚  KB Browser",
    ])

    with tab1:
        _tab_overview(checkpoint_dir, log_dir)
    with tab2:
        _tab_training_curves(checkpoint_dir, log_dir)
    with tab3:
        _tab_proof_explorer(checkpoint_dir)
    with tab4:
        _tab_kb_browser(checkpoint_dir)


if __name__ == "__main__":
    main()
