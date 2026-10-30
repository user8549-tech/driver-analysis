"""
sql_builder.py - turn the user's choices into ONE SQL string.

Pure Python: no Databricks, no Streamlit. That makes it easy to read and easy to test.

Safety rules (the notebook re-checks the SQL, but the app should never produce risky SQL in the first place):
  * identifiers (catalog, schema, table, column) are always wrapped in backticks
  * operators come from a fixed list
  * values become typed literals: numbers are parsed, text is quoted and escaped
"""
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

OPERATORS = ["=", "!=", ">", ">=", "<", "<=", "in", "is null", "is not null"]
MAX_IN_VALUES = 500

_NUMERIC_TYPES = {"tinyint", "smallint", "int", "integer", "bigint", "long", "short", "byte",
                  "float", "real", "double", "decimal", "dec", "numeric"}
_TEXT_TYPES = {"string", "varchar", "char", "date", "timestamp", "timestamp_ntz", "timestamp_ltz"}


def quote_ident(name: str) -> str:
    """`my col` -> `my col`  (a backtick inside a name is doubled)."""
    return "`" + name.replace("`", "``") + "`"


def full_name(catalog: str, schema: str, table: str) -> str:
    return ".".join(quote_ident(part) for part in (catalog, schema, table))


def column_kind(type_text: str) -> str:
    """Group Databricks column types into: numeric, boolean, text, unsupported."""
    base = re.split(r"[(<\s]", type_text.strip().lower(), maxsplit=1)[0]   # 'decimal(10,2)' -> 'decimal'
    if base in _NUMERIC_TYPES:
        return "numeric"
    if base == "boolean":
        return "boolean"
    if base in _TEXT_TYPES:          # dates / timestamps compare fine against a quoted string
        return "text"
    return "unsupported"             # array, map, struct, binary, variant, interval, ...


def to_literal(raw: str, kind: str, column: str) -> str:
    """One user-typed value -> a SQL literal of the right type."""
    if kind == "numeric":
        try:
            number = Decimal(raw)
        except InvalidOperation:
            raise ValueError(f"'{raw}' is not a number (filter on {column!r})")
        if not number.is_finite():
            raise ValueError(f"'{raw}' is not a usable number (filter on {column!r})")
        return format(number, "f")
    if kind == "boolean":
        lowered = raw.lower()
        if lowered in ("true", "1"):
            return "true"
        if lowered in ("false", "0"):
            return "false"
        raise ValueError(f"'{raw}' is not true/false (filter on {column!r})")
    # text: escape backslashes first (Spark treats them as escapes), then double the single quotes
    return "'" + raw.replace("\\", "\\\\").replace("'", "''") + "'"


def clean_filters(rows: list) -> list:
    """The filter table can contain empty rows; keep only rows where a column was chosen."""
    return [row for row in rows if row.get("column")]


def build_condition(flt: dict, column_kinds: dict) -> str:
    column, op = flt.get("column"), flt.get("op")
    if column not in column_kinds:
        raise ValueError(f"Filter uses an unknown column: {column!r}")
    if op not in OPERATORS:
        raise ValueError(f"Choose an operator for the filter on {column!r}")
    kind = column_kinds[column]
    if kind == "unsupported":
        raise ValueError(f"Filtering on {column!r} is not supported (its type is not number, text, date or boolean)")

    col = quote_ident(column)
    if op == "is null":
        return f"{col} IS NULL"
    if op == "is not null":
        return f"{col} IS NOT NULL"

    value = flt.get("value")
    raw = "" if value is None or (isinstance(value, float) and value != value) else str(value).strip()
    if raw == "":
        raise ValueError(f"The filter on {column!r} needs a value")

    if op == "in":
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts or len(parts) > MAX_IN_VALUES:
            raise ValueError(f"'in' on {column!r} needs 1-{MAX_IN_VALUES} comma-separated values")
        return f"{col} IN ({', '.join(to_literal(p, kind, column) for p in parts)})"
    return f"{col} {op} {to_literal(raw, kind, column)}"


def build_sql(table_full_name: str, filters: list, column_kinds: dict) -> str:
    """SELECT * FROM <table> [WHERE (cond1) AND (cond2) ...]"""
    conditions = [build_condition(f, column_kinds) for f in filters]
    sql = f"SELECT * FROM {table_full_name}"
    if conditions:
        sql += "\nWHERE " + "\n  AND ".join(f"({c})" for c in conditions)
    return sql


# =====================================================================================================
# Joining two tables
# =====================================================================================================
JOIN_TYPES = {
    "Inner - only rows with a match in both tables": "INNER",
    "Left - keep every row of the first table": "LEFT",
}


@dataclass
class JoinPlan:
    source_q: str      # goes after FROM:  (SELECT ... FROM a JOIN b ON ...) AS joined
    columns: list      # [{"name", "type", "origin"}]: the columns of the joined result, in order
    notes: list        # plain-English notes (which columns were left out) to show the user


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""


def clean_key_pairs(rows: list) -> list:
    """Rows from the join-columns table -> [{"left": ..., "right": ...}]. Empty rows are ignored."""
    pairs = []
    for row in rows:
        left, right = row.get("first_table_column"), row.get("second_table_column")
        if _blank(left) and _blank(right):
            continue
        if _blank(left) or _blank(right):
            raise ValueError("Each join row needs a column from BOTH tables (complete or delete the unfinished row)")
        pairs.append({"left": left, "right": right})
    return pairs


def plan_join(left_q: str, right_q: str, left_columns: list, right_columns: list, pairs: list, join_type: str) -> JoinPlan:
    """
    Build the joined table as a sub-select with UNIQUE column names (the analysis rejects duplicates).

    left_q / right_q : already-quoted table names
    *_columns        : [(name, type_text), ...]
    pairs            : [{"left": col_in_first, "right": col_in_second}, ...]  (all pairs must match)
    join_type        : "INNER" or "LEFT"

    Rules:
      * every column of the first table is kept under its own name
      * the second table's join columns are dropped (they duplicate the first table's)
      * any other second-table column whose name also exists in the first table is dropped too: the FIRST table's copy
        is the one used (names are compared case-insensitively)
      * every remaining second-table column is added after the first table's columns
    """
    if join_type not in ("INNER", "LEFT"):
        raise ValueError(f"Unsupported join type: {join_type!r}")
    if not pairs:
        raise ValueError("Add at least one pair of join columns")
    left_names, right_names = [n for n, _ in left_columns], [n for n, _ in right_columns]
    seen = set()
    for pair in pairs:
        if pair["left"] not in left_names:
            raise ValueError(f"{pair['left']!r} is not a column of the first table")
        if pair["right"] not in right_names:
            raise ValueError(f"{pair['right']!r} is not a column of the second table")
        if (pair["left"], pair["right"]) in seen:
            raise ValueError(f"The pair {pair['left']!r} = {pair['right']!r} appears twice")
        seen.add((pair["left"], pair["right"]))

    right_keys = {pair["right"] for pair in pairs}
    left_lower = {name.lower() for name in left_names}
    items = [f"l.{quote_ident(name)}" for name in left_names]
    columns = [{"name": n, "type": t, "origin": "first"} for n, t in left_columns]
    notes = [f"The second table's join column(s) {sorted(right_keys)} are left out of the result: they duplicate the first table's."]

    shared = []
    for name, type_text in right_columns:
        if name in right_keys:
            continue
        if name.lower() in left_lower:           # also in the first table: keep the first table's copy only
            shared.append(name)
            continue
        items.append(f"r.{quote_ident(name)}")
        columns.append({"name": name, "type": type_text, "origin": "second"})
    if shared:
        notes.append(f"These columns exist in both tables, so only the first table's copy is used: {shared}.")
    if len(columns) == len(left_columns):
        raise ValueError("The second table adds no new columns: everything in it is a join column "
                         "or already exists in the first table")

    on = "\n   AND ".join(f"l.{quote_ident(p['left'])} = r.{quote_ident(p['right'])}" for p in pairs)
    select = ",\n       ".join(items)
    source_q = f"(\nSELECT {select}\nFROM {left_q} AS l\n{join_type} JOIN {right_q} AS r\n  ON {on}\n) AS joined"
    return JoinPlan(source_q, columns, notes)