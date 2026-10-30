"""
app.py - the screen. Read it top to bottom: each numbered step is one small function.

  sql_builder.py    turns the user's choices into one safe SQL string      (no Databricks)
  databricks_io.py  every call to Databricks: browse, preview, run the job  (no screen code)

Page flow:  1 Choose data (one table, or two tables joined) -> 2 Choose columns -> 3 Filter rows
            -> 4 Check and run -> (run progress panel)
The run id lives in the URL (?run_id=123), so a user can leave the page and come back to the same run.

Everything after step 1 works on a `Source`: the thing the analysis will read. That is either one table, or a
sub-select that joins two tables. Steps 2-4 do not care which.
"""
import hashlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st

import databricks_io as dbx
import sql_builder as sb

st.set_page_config(page_title="Driver analysis", layout="wide")     # must be the first Streamlit call

who = dbx.current_user(st.context.headers)                           # the signed-in user (or the app, when running locally)


# =====================================================================================================
# 1. Choose data (one table, or two tables joined)
# =====================================================================================================
@dataclass
class Source:
    """What the analysis will read."""
    source_q: str             # goes after FROM: `cat`.`sch`.`tbl`   or   (SELECT ... JOIN ...) AS joined
    columns_df: pd.DataFrame  # name, type of the columns the user can pick from
    label: str                # shown in run history, e.g. "main.s.a INNER JOIN main.s.b"

    @property
    def key(self) -> str:     # makes widget keys unique per source, so changing the data resets the filter table
        return hashlib.md5(self.source_q.encode()).hexdigest()[:10]


def pick_table(title, prefix):
    """Cascading dropdowns: catalog -> schema -> table. Returns (catalog, schema, table) or None."""
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    catalog = c1.selectbox("Catalog", dbx.list_catalogs(who), index=None, placeholder="Choose a catalog",
                           key=f"{prefix}_catalog")
    schemas = dbx.list_schemas(who, catalog) if catalog else []
    schema = c2.selectbox("Schema", schemas, index=None, placeholder="Choose a schema", disabled=not catalog,
                          key=f"{prefix}_schema")
    tables = dbx.list_tables(who, catalog, schema) if schema else []
    table = c3.selectbox("Table", tables, index=None, placeholder="Choose a table", disabled=not schema,
                         key=f"{prefix}_table")
    return (catalog, schema, table) if table else None


def step1_choose_data():
    """Returns a Source, or None while the user is still choosing."""
    st.header("1. Choose data")
    first = pick_table("First table", "first")
    if first is None:
        st.info("Pick a catalog, schema and table to continue.")
        return None
    first_q = sb.full_name(*first)
    first_df = dbx.describe_columns(who, first_q)

    if not st.checkbox("Join a second table"):
        return Source(first_q, first_df, label=".".join(first))
    return joined_source(first, first_q, first_df)


def joined_source(first, first_q, first_df):
    """Pick the second table, the join type and the join columns; check the join; return a Source (or None)."""
    second = pick_table("Second table", "second")
    if second is None:
        st.info("Pick the second table, or untick 'Join a second table'.")
        return None
    second_q = sb.full_name(*second)
    second_df = dbx.describe_columns(who, second_q)

    join_label = st.radio("Join type", list(sb.JOIN_TYPES), horizontal=True)
    st.caption("Rows are matched where the chosen columns are equal. Add one row per column pair; "
               "with more than one pair, ALL pairs must match.")
    empty = pd.DataFrame({"first_table_column": pd.Series(dtype="object"), "second_table_column": pd.Series(dtype="object")})
    edited = st.data_editor(
        empty, num_rows="dynamic", hide_index=True, key=f"join_keys_{first_q}_{second_q}",
        column_config={
            "first_table_column": st.column_config.SelectboxColumn("Column in first table", options=list(first_df["name"]), required=True),
            "second_table_column": st.column_config.SelectboxColumn("Column in second table", options=list(second_df["name"]), required=True),
        })
    try:
        pairs = sb.clean_key_pairs(edited.to_dict("records"))
        if not pairs:
            st.info("Add at least one pair of join columns (use the + at the bottom of the table).")
            return None
        plan = sb.plan_join(first_q, second_q, list(zip(first_df["name"], first_df["type"])),
                            list(zip(second_df["name"], second_df["type"])), pairs, sb.JOIN_TYPES[join_label])
    except ValueError as e:
        st.error(str(e))
        return None

    if not check_join(first_q, second_q, pairs, plan, first_df, second_df):
        return None
    label = f"{'.'.join(first)} {sb.JOIN_TYPES[join_label]} JOIN {'.'.join(second)}"
    columns = pd.DataFrame(plan.columns)[["name", "type"]]
    return Source(plan.source_q, columns, label)


def check_join(first_q, second_q, pairs, plan, first_df, second_df):
    """Show what the join does and warn about risky joins. Returns False if the join cannot be used."""
    first_type = dict(zip(first_df["name"], first_df["type"]))
    second_type = dict(zip(second_df["name"], second_df["type"]))
    try:
        first_rows = dbx.count_rows(who, f"SELECT * FROM {first_q}")
        joined_rows = dbx.count_rows(who, f"SELECT * FROM {plan.source_q}")
        first_n, first_keys = dbx.key_stats(who, first_q, [p["left"] for p in pairs])
        second_n, second_keys = dbx.key_stats(who, second_q, [p["right"] for p in pairs])
    except Exception as e:
        st.error(f"The join could not run: {e}")
        return False

    st.write(f"First table: **{first_rows:,}** rows. Joined result: **{joined_rows:,}** rows.")
    for note in plan.notes:
        st.caption(note)
    if joined_rows == 0:
        st.error("The join returns no rows. Check that the join columns hold matching values.")
        return False

    for p in pairs:                                           # e.g. a number matched against text
        if sb.column_kind(first_type[p["left"]]) != sb.column_kind(second_type[p["right"]]):
            st.warning(f"{p['left']!r} ({first_type[p['left']]}) and {p['right']!r} ({second_type[p['right']]}) have different "
                       "types. Databricks converts one side before comparing, so check that the values really match.")

    first_unique, second_unique = first_n == first_keys, second_n == second_keys
    if first_unique and second_unique:
        st.caption("One-to-one join: each row matches at most one row of the other table.")
    elif second_unique:
        st.caption("Many-to-one join: several first-table rows can share one second-table row; the row count does not grow.")
    elif first_unique:
        st.info("One-to-many join: each first-table row is repeated once per matching second-table row. Rows that come from "
                "the same first-table row are not independent, so consider choosing that unit as the grouping column.")
    else:
        st.warning("Many-to-many join: BOTH tables have several rows per key, so rows multiply. This is usually a mistake; "
                   "check the join columns, or aggregate one table to one row per key first.")
    return True


def step1_overview(source):
    """Row count, columns with type and % null, and a 20-row preview. Returns the % null per column."""
    with st.spinner("Profiling the data..."):
        n_rows = dbx.count_rows(who, f"SELECT * FROM {source.source_q}")
        null_pct = dbx.null_percent(who, source.source_q, list(source.columns_df["name"]))
        sample = dbx.preview(who, source.source_q)
    st.caption(f"{n_rows:,} rows and {len(source.columns_df)} columns. (% null is measured on the first 100,000 rows.)")
    tab_cols, tab_preview = st.tabs(["Columns", "Preview (20 rows)"])
    with tab_cols:
        st.dataframe(source.columns_df.assign(percent_null=source.columns_df["name"].map(null_pct).round(1)), hide_index=True)
    with tab_preview:
        st.dataframe(sample, hide_index=True)
    return null_pct


# =====================================================================================================
# 2. Choose columns
# =====================================================================================================
def step2_choose_columns(columns):
    st.header("2. Choose columns")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target column (what you want to explain)", columns, index=None,
                          placeholder="Choose the column to explain")
    grouping = c2.selectbox("Grouping column (optional)", columns, index=None, placeholder="None - rows are independent",
                            help="Rows that belong to the same unit, e.g. the same machine, customer or battery cell. "
                                 "Leave empty if rows are independent.")
    excluded = st.multiselect("Columns to exclude (e.g. IDs)", columns)
    return target, grouping, excluded


# =====================================================================================================
# 3. Filter rows
# =====================================================================================================
def step3_filters(columns, table_key):
    """A small editable table: one row per filter. Returns a list of {column, op, value} dicts."""
    st.header("3. Filter rows (optional)")
    st.caption("Add rows with the + at the bottom of the table. For 'in', separate values with commas. "
               "Dates can be typed like 2025-01-31.")
    empty = pd.DataFrame({"column": pd.Series(dtype="object"), "op": pd.Series(dtype="object"),
                          "value": pd.Series(dtype="object")})
    edited = st.data_editor(
        empty, num_rows="dynamic", hide_index=True, key=f"filters_{table_key}",     # new table -> fresh filter table
        column_config={
            "column": st.column_config.SelectboxColumn("Column", options=list(columns), required=True),
            "op": st.column_config.SelectboxColumn("Operator", options=sb.OPERATORS, default="=", required=True),
            "value": st.column_config.TextColumn("Value"),
        })
    return sb.clean_filters(edited.to_dict("records"))


# =====================================================================================================
# 4. Check the selection and run
# =====================================================================================================
def step4_check_and_run(source, kinds, null_pct, target, grouping, excluded, filters):
    st.header("4. Check and run")
    errors, warnings = [], []

    # --- rules that only need the form (no queries)
    if target is None:
        errors.append("Choose a target column.")
    if target is not None and target == grouping:
        errors.append("The target and the grouping column must be different.")
    if target is not None and target in excluded:
        errors.append("The target column cannot also be excluded.")
    if grouping is not None and grouping in excluded:
        errors.append("The grouping column cannot also be excluded.")
    if target is not None and null_pct[target] > 30:
        warnings.append(f"The target is {null_pct[target]:.0f}% null (first 100,000 rows); those rows will be dropped.")

    # --- build the SQL, then rules that need the data (queries are cached, so reruns are cheap)
    sql = None
    try:
        sql = sb.build_sql(source.source_q, filters, kinds)
    except ValueError as e:
        errors.append(str(e))

    if sql:
        try:
            n_rows = dbx.count_rows(who, sql)
            st.write(f"**{n_rows:,}** rows match your filters.")
            if n_rows == 0:
                errors.append("No rows match the filters.")
            elif n_rows < 50:
                warnings.append(f"Only {n_rows} rows match; results will be unreliable.")
            if target is not None and kinds[target] != "numeric" and dbx.distinct_count(who, sql, target) != 2:
                errors.append("The target must be numeric or have exactly two distinct values.")
            if grouping is not None and dbx.distinct_count(who, sql, grouping) < 8:
                warnings.append("The grouping column has fewer than 8 distinct values; group-based checks will be weak.")
        except Exception as e:
            errors.append(f"The query could not run: {e}")
        with st.expander("SQL that will be sent to the analysis"):
            st.code(sql, language="sql")

    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)

    if st.button("Run analysis", type="primary", disabled=bool(errors)):
        try:
            dbx.check_user_can_run(who, sql)             # access check as the USER; the job itself runs as the app
        except Exception as e:
            st.error(f"You don't seem to have access to run this query: {e}")
            return
        label = source.label + (f" ({len(filters)} filter{'s' if len(filters) != 1 else ''})" if filters else "")
        try:
            run_id = dbx.start_run(sql, target, grouping, excluded, config={}, run_label=label)
        except Exception as e:
            st.error(f"Could not start the job: {e}")
            return
        st.query_params["run_id"] = str(run_id)          # put the run in the URL, then redraw as the progress page
        st.rerun()


# =====================================================================================================
# Run progress (shown instead of the form once a run exists)
# =====================================================================================================
def current_run_id():
    raw = st.query_params.get("run_id")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def render_status(run_id, status):
    labels = {"queued": "Queued / starting", "running": "Running", "succeeded": "Finished",
              "failed": "Failed", "cancelled": "Cancelled"}
    minutes, seconds = divmod(status.elapsed_s, 60)
    st.subheader(f"{labels[status.state]}  -  {minutes}:{seconds:02d} elapsed")
    if status.detail:
        st.caption(status.detail)
    if status.page_url:
        st.link_button("Open this run in Databricks", status.page_url)
    if status.is_active and st.button("Cancel run", key=f"cancel_{run_id}"):
        dbx.cancel_run(run_id)
        st.rerun()
    if status.state == "failed":
        with st.expander("Error details", expanded=True):
            st.code(status.error or "(no error text)")
    if status.state == "succeeded":
        st.success("The analysis finished. Next step (Prompt 3): read results.json from the folder below and show it here.")
        st.code(status.output or "(the notebook returned no output)", language="json")


@st.fragment(run_every=3)                                # re-runs only this block every 3 seconds
def live_run_panel(run_id):
    status = dbx.get_run_status(run_id)
    if not status.is_active:
        st.rerun()                                       # just finished: redraw the whole page so polling stops
    render_status(run_id, status)


def show_run(run_id):
    st.header(f"Analysis run {run_id}")
    if st.button("Start a new analysis"):
        st.query_params.clear()
        st.rerun()
    try:
        status = dbx.get_run_status(run_id)
    except Exception as e:
        st.error(f"Could not read run {run_id}: {e}")
        return
    if status.is_active:
        live_run_panel(run_id)
    else:
        render_status(run_id, status)


# =====================================================================================================
# The page
# =====================================================================================================
def main():
    st.title("Driver analysis")
    with st.sidebar:
        if who.on_behalf_of_user:
            st.success(f"Signed in as {who.name}")
        else:
            st.warning("No user token found, so browsing and the access check use the app's own identity. "
                       "Expected when running locally; in the deployed app it means user authorization is off.")

    run_id = current_run_id()
    if run_id:
        show_run(run_id)
        return

    source = step1_choose_data()
    if source is None:
        return
    columns = list(source.columns_df["name"])
    kinds = {name: sb.column_kind(t) for name, t in zip(source.columns_df["name"], source.columns_df["type"])}

    null_pct = step1_overview(source)
    target, grouping, excluded = step2_choose_columns(columns)
    filters = step3_filters(columns, table_key=source.key)
    step4_check_and_run(source, kinds, null_pct, target, grouping, excluded, filters)


main()