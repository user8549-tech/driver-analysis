"""
app.py - the screen. Read it top to bottom: each numbered step is one small function.

  sql_builder.py    turns the user's choices into one safe SQL string      (no Databricks)
  databricks_io.py  every call to Databricks: browse, preview, run the job  (no screen code)

Page flow:  1 Choose data -> 2 Choose columns -> 3 Filter rows -> 4 Check and run -> (run progress panel)
The run id lives in the URL (?run_id=123), so a user can leave the page and come back to the same run.
"""
import pandas as pd
import streamlit as st


import databricks_io as dbx
import sql_builder as sb

st.set_page_config(page_title="Driver analysis", layout="wide")     # must be the first Streamlit call

who = dbx.current_user(st.context.headers)                           # the signed-in user (or the app, when running locally)


# =====================================================================================================
# 1. Choose data
# =====================================================================================================
def step1_choose_table():
    """Cascading dropdowns: catalog -> schema -> table. Returns (catalog, schema, table) or None."""
    st.header("1. Choose data")
    c1, c2, c3 = st.columns(3)
    catalog = c1.selectbox("Catalog", dbx.list_catalogs(who), index=None, placeholder="Choose a catalog")
    schemas = dbx.list_schemas(who, catalog) if catalog else []
    schema = c2.selectbox("Schema", schemas, index=None, placeholder="Choose a schema", disabled=not catalog)
    tables = dbx.list_tables(who, catalog, schema) if schema else []
    table = c3.selectbox("Table", tables, index=None, placeholder="Choose a table", disabled=not schema)
    if not table:
        st.info("Pick a catalog, schema and table to continue.")
        return None
    return catalog, schema, table


def step1_overview(table_q, columns_df):
    """Row count, columns with type and % null, and a 20-row preview. Returns the % null per column."""
    with st.spinner("Profiling the table..."):
        n_rows = dbx.count_rows(who, f"SELECT * FROM {table_q}")
        null_pct = dbx.null_percent(who, table_q, list(columns_df["name"]))
        sample = dbx.preview(who, table_q)
    st.caption(f"{n_rows:,} rows and {len(columns_df)} columns. (% null is measured on the first 100,000 rows.)")
    tab_cols, tab_preview = st.tabs(["Columns", "Preview (20 rows)"])
    with tab_cols:
        st.dataframe(columns_df.assign(percent_null=columns_df["name"].map(null_pct).round(1)), hide_index=True)
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
def step4_check_and_run(table_parts, table_q, kinds, null_pct, target, grouping, excluded, filters):
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
        sql = sb.build_sql(table_q, filters, kinds)
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
        label = ".".join(table_parts) + (f" ({len(filters)} filter{'s' if len(filters) != 1 else ''})" if filters else "")
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

    picked = step1_choose_table()
    if picked is None:
        return
    catalog, schema, table = picked
    table_q = sb.full_name(catalog, schema, table)
    columns_df = dbx.describe_columns(who, table_q)
    columns = list(columns_df["name"])
    kinds = {name: sb.column_kind(t) for name, t in zip(columns_df["name"], columns_df["type"])}

    null_pct = step1_overview(table_q, columns_df)
    target, grouping, excluded = step2_choose_columns(columns)
    filters = step3_filters(columns, table_key=table_q)
    step4_check_and_run((catalog, schema, table), table_q, kinds, null_pct, target, grouping, excluded, filters)


main()