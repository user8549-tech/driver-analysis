"""
results_view.py - shows the results of a finished analysis (Prompt 3).

Two halves, read in this order:
  A. PURE HELPERS   number formatting, building / filtering / colouring the tables.  No Streamlit, easy to test.
  B. SCREEN         one function per block of the page, in the order the user sees them:
                    banner -> header card -> warnings -> drivers table -> model comparison -> cleaning log
                    plus a second tab with the full notebook report.

The data comes from results.json (written by the notebook). Every field is optional: a missing field is shown as
"-" or the block is skipped. It must never crash the page.

Two entry points, ONE page: show_results() for a run that just finished, and show_past_results() for results saved by
an earlier run. Both end up in show_results_page(), so imported results look exactly like fresh ones.
"""
import html
import json
import math

import pandas as pd
import streamlit as st

import databricks_io as dbx

try:                                    # Styler.background_gradient (the colour scales) needs matplotlib
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BANNER_TEXT = "Model does not predict unseen groups - importances are not trustworthy"


# =====================================================================================================
# A. PURE HELPERS
# =====================================================================================================
def num(value):
    """A real number, or None (None, NaN, inf, text and booleans all become None)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt(value, spec=".3f"):
    x = num(value)
    return "-" if x is None else format(x, spec)


def fmt_int(value):
    x = num(value)
    return "-" if x is None else f"{int(round(x)):,}"


def as_dict(value):
    return value if isinstance(value, dict) else {}


def as_list(value):
    return value if isinstance(value, list) else []


def results_path_from_output(output):
    """The notebook's exit value is a small JSON string like {"status": "ok", "results_path": "/Volumes/..."}."""
    try:
        data = json.loads(output or "")
    except (TypeError, ValueError):
        return None
    path = data.get("results_path") if isinstance(data, dict) else None
    return path if isinstance(path, str) and path.strip() else None


def mode_text(meta):
    small = meta.get("small_data")
    mode = "Small-data" if small is True else ("Normal" if small is False else None)
    task = meta.get("task_type")
    return " · ".join(str(x) for x in (mode, task) if x) or "-"


# ---------- the drivers table ----------
DRIVER_COLUMN_ORDER = [
    "feature", "cluster", "importance_mean", "importance_std", "rank_mean", "rank_std", "groups_hurt",
    "partial_cv", "standalone_cv", "effect_size", "fdr_q", "fdr_q_group_level", "within_group_r2",
    "group_level", "direction", "drop_single", "drop_cluster", "max_vif",
]
TEXT_COLUMNS = {"feature", "cluster", "direction", "model"}
BOOL_COLUMNS = {"group_level"}
GOOD_HIGH = {"importance_mean", "groups_hurt", "partial_cv", "standalone_cv", "effect_size", "within_group_r2",
             "drop_single", "drop_cluster", "pooled_score", "groups_where_better_than_mean_only"}   # bigger = stronger evidence
GOOD_LOW = {"rank_mean", "rank_std", "fdr_q", "fdr_q_group_level", "max_vif"}                     # smaller = stronger evidence
ZERO_TO_ONE = {"groups_hurt", "fdr_q", "fdr_q_group_level", "within_group_r2", "effect_size",
               "groups_where_better_than_mean_only"}                                                # fixed colour range

COLUMN_HELP = {
    "cluster": "Group of redundant (highly associated) features. Credit the cluster, not one member.",
    "importance_mean": "How much the model's score drops when this feature is shuffled. Bigger = the model relies on it more.",
    "importance_std": "Spread of that drop across repeats / splits. It does not include the uncertainty from having few groups.",
    "rank_mean": "Average importance rank (1 = most important).",
    "rank_std": "How much the rank moves between repeats / splits. Above about 2 means the ordering is unstable.",
    "groups_hurt": "Share of held-out groups whose predictions got worse when the feature was shuffled. Want above 0.7 (small-data mode).",
    "partial_cv": "How much the cross-validated score of the linear model falls when this feature is removed.",
    "standalone_cv": "Cross-validated score of a linear model that uses only this feature.",
    "effect_size": "Strength of this feature's one-to-one relationship with the target (0 to 1).",
    "fdr_q": "Row-level significance after correcting for many tests. Smaller = stronger. Optimistic when rows within a group are alike.",
    "fdr_q_group_level": "The same test with one data point per group. The honest version when groups matter.",
    "within_group_r2": "Share of the target's variation inside groups explained by this numeric feature.",
    "group_level": "True if the feature does not change inside a group. Judge it with fdr_q_group_level, not fdr_q.",
    "direction": "Numeric features: + or - relationship. Categories: which level has the highest / lowest target average.",
    "drop_single": "Drop in cross-validated score when this one feature is removed.",
    "drop_cluster": "Drop in cross-validated score when this feature's whole cluster is removed.",
    "max_vif": "Collinearity. Above 10 means the feature is nearly a combination of the others.",
}


def _coerce(df):
    """Numbers become numbers (bad values -> NaN), text stays text, booleans stay booleans."""
    for col in df.columns:
        if col in TEXT_COLUMNS:
            df[col] = df[col].astype("object")
        elif col in BOOL_COLUMNS:
            df[col] = df[col].map(lambda v: v if isinstance(v, bool) else None).astype("object")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drivers_frame(summary):
    """summary (list of dicts) -> DataFrame: sensible column order, sorted by importance (biggest first)."""
    rows = [r for r in as_list(summary) if isinstance(r, dict)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "feature" not in df.columns:
        df.insert(0, "feature", [f"row {i + 1}" for i in range(len(df))])
    df = _coerce(df)
    df = df[[c for c in DRIVER_COLUMN_ORDER if c in df.columns] + [c for c in df.columns if c not in DRIVER_COLUMN_ORDER]]
    if "importance_mean" in df.columns:
        df = df.sort_values("importance_mean", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def filter_drivers(df, search="", clusters=None, only_positive=False):
    """Name search (ignores case), cluster selection, and 'importance above zero only'."""
    out = df
    if (search or "").strip():
        out = out[out["feature"].astype(str).str.contains(search.strip(), case=False, regex=False, na=False)]
    if clusters and "cluster" in out.columns:
        out = out[out["cluster"].isin(list(clusters))]
    if only_positive and "importance_mean" in out.columns:
        out = out[out["importance_mean"] > 0]
    return out


def style_table(df):
    """3 decimals, '-' for anything missing, and a colour scale on every numeric column (green = stronger evidence)."""
    df = df.reset_index(drop=True)                                     # so row labels are simply 0, 1, 2, ...
    numeric = [c for c in df.columns if c not in TEXT_COLUMNS and c not in BOOL_COLUMNS]
    styler = df.style.hide(axis="index").format(na_rep="-")            # missing text / booleans show '-', not 'nan' / 'None'
    if numeric:
        styler = styler.format("{:.3f}", subset=numeric, na_rep="-")
    if HAS_MATPLOTLIB:
        for col in numeric:
            present = df.index[df[col].notna()]                        # colour ONLY cells that have a value:
            if len(present) < 2:                                       # (matplotlib would paint a missing one black)
                continue                                               # and a scale needs at least two values
            cmap = "Greens" if col in GOOD_HIGH else ("Greens_r" if col in GOOD_LOW else "Blues")
            vmin, vmax = (0, 1) if col in ZERO_TO_ONE else (None, None)
            styler = styler.background_gradient(cmap=cmap, subset=pd.IndexSlice[present, col], vmin=vmin, vmax=vmax)
    return styler


def comparison_frame(records):
    rows = [r for r in as_list(records) if isinstance(r, dict)]
    if not rows:
        return pd.DataFrame()
    df = _coerce(pd.DataFrame(rows))
    return df[["model"] + [c for c in df.columns if c != "model"]] if "model" in df.columns else df


# ---------- the list of past analyses ----------
def past_run_label(run):
    """One line that identifies a saved analysis in a dropdown (the folder name keeps it unique)."""
    return " · ".join([created_text(run.get("created_utc")), str(run.get("label") or "(no label)"),
                       str(run.get("target") or "-"), str(run.get("folder"))])


def created_text(value):
    """'2026-10-29T23:34:49+00:00' -> '2026-10-29 23:34'. Anything unexpected -> '-'."""
    text = str(value or "").strip()
    return text[:16].replace("T", " ") if len(text) >= 16 else "-"


def past_runs_frame(runs):
    """List of run summaries (from databricks_io.list_past_runs) -> a tidy table for display."""
    rows = []
    for run in as_list(runs):
        if not isinstance(run, dict):
            continue
        score = num(run.get("val_score"))
        small = run.get("small_data")
        rows.append({
            "Created": created_text(run.get("created_utc")),
            "Label": run.get("label") or "-",
            "Target": run.get("target") or "-",
            "Mode": "Small-data" if small is True else ("Normal" if small is False else "-"),
            "Rows": fmt_int(run.get("rows_used")),
            "Groups": fmt_int(run.get("n_groups")),
            "Score": "-" if score is None else f"{run.get('score_name') or 'score'} {score:.3f}",
            "Gate": {True: "passed", False: "FAILED"}.get(run.get("model_ok"), "-"),
            "Folder": str(run.get("folder", "-")),
        })
    return pd.DataFrame(rows)


def sandboxed_iframe(report_html, height=850):
    """
    The exported notebook is HTML that needs its own scripts to draw itself, so it runs inside an iframe with
    sandbox="allow-scripts": scripts may run, but WITHOUT access to this app's page, cookies or storage
    (no allow-same-origin), and without forms, popups or navigation.
    """
    return (f'<body style="margin:0"><iframe sandbox="allow-scripts" srcdoc="{html.escape(report_html, quote=True)}" '
            f'style="width:100%;height:{height}px;border:1px solid #ccc;border-radius:4px"></iframe></body>')


# =====================================================================================================
# B. SCREEN
# =====================================================================================================
def show_results(run_id, status):
    """Entry point 1: a run that just finished. The notebook's exit value says where its results.json is."""
    results_path = results_path_from_output(status.output)
    if results_path is None:
        st.warning("The run finished, but its output does not say where the results are (no results_path).")
        st.code(status.output or "(the notebook returned no output)", language="json")
        return
    show_results_page(results_path, key=str(run_id), run_id=run_id, raw_output=status.output)


def show_past_results(name):
    """Entry point 2: results saved by an earlier run, opened by the name of their folder under RESULTS_ROOT."""
    try:
        results_path = dbx.past_results_path(name)
    except ValueError as e:
        st.error(str(e))
        return
    # A folder named after a job run id can also show that run's notebook report (while Databricks still keeps the run).
    run_id = int(name) if name.strip().isdigit() else None
    show_results_page(results_path, key=f"past_{name.strip()}", run_id=run_id, raw_output=None, imported=True)


def show_results_page(results_path, key, run_id=None, raw_output=None, imported=False):
    """
    The results page. `key` keeps this page's widgets apart from other pages'. `run_id` (optional) enables the
    Full report tab. `imported` adds a note saying these are saved results from an earlier run.
    """
    try:
        results = dbx.read_results(results_path)
    except Exception as e:
        what = "The run finished, but its results" if raw_output else "The results"
        st.error(f"{what} could not be read from {results_path}: {e}")
        if raw_output:
            st.code(raw_output, language="json")
        return

    meta = as_dict(results.get("meta"))
    if imported:
        when, label = created_text(meta.get("created_utc")), meta.get("label")
        st.info(f"These are saved results from an earlier run (created {when}" + (f", {label}" if label else "") + ").")
    tab_results, tab_report = st.tabs(["Results", "Full report"])
    with tab_results:
        show_banner(meta)
        show_header_card(meta)
        show_warnings(as_list(results.get("warnings")))
        show_drivers(key, as_list(results.get("summary")))
        show_model_comparison(results.get("model_comparison"))
        show_cleaning_log(as_list(results.get("clean_log")))
    with tab_report:
        if run_id is None:
            st.info("The full notebook report is only available for results that came from a job run "
                    "(a results folder named after the run id).")
        else:
            show_full_report(run_id)
    with st.expander("Technical details"):
        st.write(f"Results folder: {results_path}")
        if raw_output:
            st.code(raw_output, language="json")
        if meta.get("label"):
            st.write(f"Label: {meta['label']}  |  Created: {meta.get('created_utc', '-')}")
        if meta.get("sql_query"):
            st.code(meta["sql_query"], language="sql")


def show_banner(meta):
    model_ok = meta.get("model_ok")
    if model_ok is False:
        st.error(f"**{BANNER_TEXT}**", icon="🚫")
    elif model_ok is True:
        st.success("The model beats the baseline on unseen groups (it passes the validation gate).")
    else:
        st.info("This report has no validation-gate result, so the importances cannot be judged.")


def show_header_card(meta):
    score, baseline = num(meta.get("val_score")), num(meta.get("baseline_score"))
    delta = f"{score - baseline:+.3f} vs baseline {baseline:.3f}" if score is not None and baseline is not None else None
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Target", str(meta.get("target") or "-"))
        c2.metric("Mode", mode_text(meta))
        c3.metric("Rows used", fmt_int(meta.get("rows_used")))
        c4.metric("Groups", fmt_int(meta.get("n_groups")))
        c5.metric(str(meta.get("score_name") or "Score"), fmt(score), delta=delta)
        notes = []
        if meta.get("eval_label"):
            notes.append(f"Evaluation: {meta['eval_label']}")
        if num(meta.get("gate")) is not None:
            notes.append(f"validation gate: {fmt(meta['gate'], '.2f')}")
        if num(meta.get("icc")) is not None:
            notes.append(f"ICC: {fmt(meta['icc'], '.2f')} (share of the target's variance that lies between groups)")
        if notes:
            st.caption(" · ".join(notes))


def show_warnings(warnings):
    st.subheader(f"Warnings ({len(warnings)})" if warnings else "Warnings")
    if not warnings:
        st.caption("No warnings.")
        return
    for text in warnings:
        st.warning(str(text))


def show_drivers(key, summary):
    st.subheader("Drivers")
    df = drivers_frame(summary)
    if df.empty:
        st.info("This report has no per-feature results.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    search = c1.text_input("Search features", key=f"drivers_search_{key}")
    clusters = sorted(df["cluster"].dropna().astype(str).unique()) if "cluster" in df.columns else []
    chosen = c2.multiselect("Cluster", clusters, key=f"drivers_clusters_{key}") if clusters else []
    only_positive = c3.checkbox("Importance > 0 only", key=f"drivers_pos_{key}") if "importance_mean" in df.columns else False

    shown = filter_drivers(df, search, chosen, only_positive)
    st.caption(f"Showing {len(shown)} of {len(df)} features. Click a column header to sort. Green = stronger evidence.")
    if shown.empty:
        st.info("No features match the filters.")
    else:
        st.dataframe(style_table(shown), hide_index=True)
    with st.expander("What do these columns mean?"):
        for col in df.columns:
            if col in COLUMN_HELP:
                st.markdown(f"**{col}**: {COLUMN_HELP[col]}")


def show_model_comparison(records):
    df = comparison_frame(records)
    if df.empty:
        return
    st.subheader("Model comparison")
    st.caption("Every row of data was predicted by a model that never saw its group. 'mean-only' is the baseline to beat.")
    st.dataframe(style_table(df), hide_index=True)


def show_cleaning_log(log):
    if not log:
        return
    with st.expander(f"What the analysis did to your data ({len(log)} notes)"):
        st.text("\n".join(f"- {line}" for line in log))


def show_full_report(run_id):
    """The notebook exactly as it ran. Loaded on demand: the export can be several MB."""
    st.caption("The notebook as it ran: every table and chart. Loaded on request because it can be large.")
    flag = f"report_loaded_{run_id}"
    if st.button("Load full report", key=f"load_{flag}"):
        st.session_state[flag] = True
    if not st.session_state.get(flag):
        return
    try:
        with st.spinner("Exporting the notebook run..."):
            report_html = dbx.export_run_html(run_id)
    except Exception as e:
        st.error(f"The report could not be exported: {e}")
        return
    st.download_button("Download HTML", data=report_html, file_name=f"driver-analysis-run-{run_id}.html",
                       mime="text/html", key=f"download_{run_id}")
    st.components.v1.html(sandboxed_iframe(report_html), height=870, scrolling=False)