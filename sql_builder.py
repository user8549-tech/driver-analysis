"""
sql_builder.py - turn the user's choices into ONE SQL string.

Pure Python: no Databricks, no Streamlit. That makes it easy to read and easy to test.

Safety rules (the notebook re-checks the SQL, but the app should never produce risky SQL in the first place):
  * identifiers (catalog, schema, table, column) are always wrapped in backticks
  * operators come from a fixed list
  * values become typed literals: numbers are parsed, text is quoted and escaped
"""
import re
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