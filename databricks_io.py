"""
databricks_io.py - EVERY call to Databricks lives in this file. app.py never touches the SDK directly.

Two identities are in play:
  * the USER   (their token arrives in a request header)  -> browse tables, preview data, check access
  * the APP    (the app's own service principal)          -> start the job, poll it, cancel it, write long SQL files
"""
import io
import json
import os
import time
import uuid
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.sql import StatementState

from sql_builder import quote_ident

INLINE_SQL_LIMIT = 8_000    # bytes. Longer SQL is written to a file because job parameters are size-limited.


# =====================================================================================================
# 1. Who is using the app?
# =====================================================================================================
@dataclass
class Who:
    name: str                 # also used in cache keys, so one user never sees another user's cached results
    client: WorkspaceClient   # the identity used for browsing / previewing / access checks
    on_behalf_of_user: bool   # False = fell back to the app's own identity


@st.cache_resource
def app_client() -> WorkspaceClient:
    """The app's own identity (service principal). Credentials are injected by Databricks Apps."""
    return WorkspaceClient()


def current_user(headers) -> Who:
    token = headers.get("x-forwarded-access-token")      # only present in the deployed app, with user authorization on
    if not token:
        return Who("app-service-principal", app_client(), False)
    name = headers.get("x-forwarded-email") or headers.get("x-forwarded-preferred-username") or "unknown-user"
    client = WorkspaceClient(host=Config().host, token=token, auth_type="pat")
    return Who(name, client, True)


# =====================================================================================================
# 2. Run SQL on the SQL warehouse (used for browsing too: SHOW CATALOGS, DESCRIBE TABLE, ...)
# =====================================================================================================
def run_sql(w: WorkspaceClient, statement: str, timeout_s: int = 120) -> pd.DataFrame:
    """Run one statement, wait for it, return the rows as a DataFrame of strings."""
    api = w.statement_execution
    resp = api.execute_statement(statement=statement, warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"],
                                 wait_timeout="30s")
    started = time.time()
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - started > timeout_s:
            api.cancel_execution(resp.statement_id)
            raise TimeoutError(f"The query took longer than {timeout_s}s and was cancelled")
        time.sleep(1)
        resp = api.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(resp.status.error.message if resp.status.error else f"Query ended as {resp.status.state}")
    columns = [c.name for c in resp.manifest.schema.columns]
    rows = resp.result.data_array if resp.result and resp.result.data_array else []
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(ttl=300, show_spinner=False)
def query(user: str, _w, statement: str) -> pd.DataFrame:
    """run_sql with a 5-minute cache. `user` is part of the cache key; `_w` (leading underscore) is not hashed."""
    return run_sql(_w, statement)


# =====================================================================================================
# 3. Browse Unity Catalog (all via SQL, so the user's own permissions decide what they can see)
# =====================================================================================================
def list_catalogs(who: Who) -> list:
    return sorted(query(who.name, who.client, "SHOW CATALOGS").iloc[:, 0])


def list_schemas(who: Who, catalog: str) -> list:
    names = query(who.name, who.client, f"SHOW SCHEMAS IN {quote_ident(catalog)}").iloc[:, 0]
    return sorted(n for n in names if n != "information_schema")


def list_tables(who: Who, catalog: str, schema: str) -> list:
    df = query(who.name, who.client, f"SHOW TABLES IN {quote_ident(catalog)}.{quote_ident(schema)}")
    return sorted(df.iloc[:, 1])                           # columns are: database, tableName, isTemporary


def describe_columns(who: Who, table_q: str) -> pd.DataFrame:
    """Returns a DataFrame with columns: name, type. `table_q` is the already-quoted table name."""
    df = query(who.name, who.client, f"DESCRIBE TABLE {table_q}")
    names, types = [], []
    for col_name, data_type in zip(df.iloc[:, 0], df.iloc[:, 1]):
        if not col_name or str(col_name).startswith("#"):  # DESCRIBE adds partition / metadata sections after a blank row
            break
        names.append(col_name); types.append(data_type)
    return pd.DataFrame({"name": names, "type": types})


# =====================================================================================================
# 4. Profile the data the user picked
# =====================================================================================================
def count_rows(who: Who, sql: str) -> int:
    return int(query(who.name, who.client, f"SELECT count(*) AS n FROM (\n{sql}\n) AS t").iloc[0, 0])


def distinct_count(who: Who, sql: str, column: str) -> int:
    stmt = f"SELECT count(DISTINCT {quote_ident(column)}) AS n FROM (\n{sql}\n) AS t"
    return int(query(who.name, who.client, stmt).iloc[0, 0])


def preview(who: Who, table_q: str, n: int = 20) -> pd.DataFrame:
    return query(who.name, who.client, f"SELECT * FROM {table_q} LIMIT {n}")


def null_percent(who: Who, table_q: str, columns: list, sample_rows: int = 100_000) -> pd.Series:
    """% of NULLs per column, measured on the first `sample_rows` rows (approximate on big tables)."""
    counts = ", ".join(f"count({quote_ident(c)}) AS c{i}" for i, c in enumerate(columns))
    stmt = f"SELECT count(*) AS n, {counts} FROM (SELECT * FROM {table_q} LIMIT {sample_rows}) AS s"
    row = query(who.name, who.client, stmt).iloc[0]
    n = int(row["n"])
    return pd.Series({c: (0.0 if n == 0 else 100.0 * (1 - int(row[f"c{i}"]) / n)) for i, c in enumerate(columns)})


def key_stats(who: Who, table_q: str, columns: list) -> tuple:
    """For a join: (rows, distinct_keys) among rows where every key column is non-null (null keys never match).
    rows == distinct_keys means each key value appears once in this table."""
    cols = ", ".join(quote_ident(c) for c in columns)
    not_null = " AND ".join(f"{quote_ident(c)} IS NOT NULL" for c in columns)
    stmt = f"SELECT count(*) AS n, count(DISTINCT struct({cols})) AS d FROM {table_q} WHERE {not_null}"
    row = query(who.name, who.client, stmt).iloc[0]
    return int(row["n"]), int(row["d"])


def check_user_can_run(who: Who, sql: str) -> None:
    """Run the final SQL with LIMIT 0 as the USER (not cached). Raises if they lack access or the SQL is invalid."""
    run_sql(who.client, f"SELECT * FROM (\n{sql}\n) AS t LIMIT 0")


# =====================================================================================================
# 5. Start, watch and cancel the analysis job (always as the APP identity)
# =====================================================================================================
def start_run(sql: str, target: str, grouping, exclude_cols: list, config: dict, run_label: str) -> int:
    """Start the existing Job. Returns the run id. The job's parameter names must match the notebook wrapper."""
    w = app_client()
    results_root = os.environ["RESULTS_ROOT"]
    params = {
        "sql_query": sql,
        "sql_query_path": "",
        "target_variable": target,
        "grouping_var": grouping or "",
        "exclude_cols": ",".join(exclude_cols),
        "config_json": json.dumps(config or {}),
        "run_label": run_label[:200],
        "results_root": results_root,
    }
    if len(sql.encode("utf-8")) > INLINE_SQL_LIMIT:        # too long for a parameter: hand over a file instead
        path = f"{results_root.rstrip('/')}/inputs/{uuid.uuid4().hex}.sql"
        w.files.upload(path, io.BytesIO(sql.encode("utf-8")), overwrite=True)
        params["sql_query"], params["sql_query_path"] = "", path
    waiter = w.jobs.run_now(job_id=int(os.environ["ANALYSIS_JOB_ID"]), job_parameters=params)
    return waiter.response.run_id


@dataclass
class RunStatus:
    state: str            # queued | running | succeeded | failed | cancelled
    detail: str           # Databricks' own state message
    elapsed_s: int
    page_url: str         # link to the run in the Databricks UI
    output: str = ""      # on success: the notebook's exit value (JSON with results_path)
    error: str = ""       # on failure: the error text

    @property
    def is_active(self) -> bool:
        return self.state in ("queued", "running")


def get_run_status(run_id: int) -> RunStatus:
    w = app_client()
    run = w.jobs.get_run(run_id=run_id)
    life = getattr(run.state.life_cycle_state, "value", "") if run.state else ""
    result = getattr(run.state.result_state, "value", "") if run.state else ""
    detail = (run.state.state_message if run.state else "") or ""

    if life in ("QUEUED", "PENDING", "BLOCKED", "WAITING_FOR_RETRY"):
        state = "queued"
    elif life in ("RUNNING", "TERMINATING"):
        state = "running"
    elif life == "TERMINATED":
        state = "succeeded" if result == "SUCCESS" else ("cancelled" if result == "CANCELED" else "failed")
    else:                                                   # SKIPPED, INTERNAL_ERROR, anything unknown
        state = "failed"

    start = run.start_time or 0
    end = run.end_time or int(time.time() * 1000)
    status = RunStatus(state, detail, int(max(0, end - start) / 1000) if start else 0, run.run_page_url or "")

    if state in ("succeeded", "failed") and run.tasks:      # the notebook's own output / error lives on the task run
        try:
            out = w.jobs.get_run_output(run_id=run.tasks[0].run_id)
            if state == "succeeded":
                status.output = (out.notebook_output.result if out.notebook_output else "") or ""
            else:
                status.error = "\n".join(x for x in (out.error, out.error_trace) if x)
        except Exception as e:                              # the run status is still useful without the details
            status.error = status.error or f"(could not fetch the error details: {e})"
    if state == "failed" and not status.error:
        status.error = detail
    return status


def cancel_run(run_id: int) -> None:
    app_client().jobs.cancel_run(run_id=run_id)