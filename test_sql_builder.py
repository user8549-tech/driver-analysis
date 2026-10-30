"""Run with:  python test_sql_builder.py   (or: pytest)"""
import sql_builder as sb

KINDS = {"y": "numeric", "name": "text", "when": "text", "ok": "boolean", "tags": "unsupported", "we`ird col": "text"}
T = sb.full_name("main", "sandbox", "batteries")


def expect_error(fn, fragment):
    try:
        fn()
    except ValueError as e:
        assert fragment in str(e), f"wrong message: {e}"
        return
    raise AssertionError("expected ValueError")


def test_names_and_kinds():
    assert T == "`main`.`sandbox`.`batteries`"
    assert sb.quote_ident("we`ird") == "`we``ird`"
    assert sb.column_kind("decimal(10,2)") == "numeric" and sb.column_kind("BIGINT") == "numeric"
    assert sb.column_kind("string") == "text" and sb.column_kind("varchar(20)") == "text" and sb.column_kind("char(3)") == "text"
    assert sb.column_kind("timestamp") == "datetime" and sb.column_kind("date") == "datetime" and sb.column_kind("TIMESTAMP_NTZ") == "datetime"
    assert sb.column_kind("boolean") == "boolean"
    assert sb.column_kind("interval day") == "unsupported"       # must not be mistaken for 'int'
    assert sb.column_kind("array<string>") == "unsupported" and sb.column_kind("struct<a:int>") == "unsupported"


def test_basic_sql():
    assert sb.build_sql(T, [], KINDS) == "SELECT * FROM `main`.`sandbox`.`batteries`"
    sql = sb.build_sql(T, [{"column": "y", "op": ">=", "value": "5"}, {"column": "name", "op": "in", "value": "a, b"}], KINDS)
    assert sql == ("SELECT * FROM `main`.`sandbox`.`batteries`\nWHERE (`y` >= 5)\n  AND (`name` IN ('a', 'b'))")


def test_literals_are_escaped():
    assert sb.to_literal("it's", "text", "name") == "'it''s'"
    assert sb.to_literal("a\\b", "text", "name") == "'a\\\\b'"
    evil = "x'; DROP TABLE t; --"
    cond = sb.build_condition({"column": "name", "op": "=", "value": evil}, KINDS)
    assert cond == "`name` = 'x''; DROP TABLE t; --'"              # stays inside the quotes
    assert sb.build_condition({"column": "we`ird col", "op": "is null"}, KINDS) == "`we``ird col` IS NULL"


def test_numbers_and_booleans_are_parsed_not_pasted():
    assert sb.to_literal("1e3", "numeric", "y") == "1000"
    assert sb.to_literal("-2.50", "numeric", "y") == "-2.50"
    assert sb.to_literal("007", "numeric", "y") == "7"
    expect_error(lambda: sb.to_literal("5; DROP TABLE t", "numeric", "y"), "not a number")
    expect_error(lambda: sb.to_literal("NaN", "numeric", "y"), "not a usable number")
    expect_error(lambda: sb.to_literal("Infinity", "numeric", "y"), "not a usable number")
    assert sb.to_literal("TRUE", "boolean", "ok") == "true"
    expect_error(lambda: sb.to_literal("maybe", "boolean", "ok"), "not true/false")


def test_validation_errors():
    expect_error(lambda: sb.build_condition({"column": "nope", "op": "="}, KINDS), "unknown column")
    expect_error(lambda: sb.build_condition({"column": "y", "op": "like", "value": "1"}, KINDS), "operator")
    expect_error(lambda: sb.build_condition({"column": "y", "op": None, "value": "1"}, KINDS), "operator")
    expect_error(lambda: sb.build_condition({"column": "y", "op": ">", "value": ""}, KINDS), "needs a value")
    expect_error(lambda: sb.build_condition({"column": "y", "op": ">", "value": float("nan")}, KINDS), "needs a value")
    expect_error(lambda: sb.build_condition({"column": "tags", "op": "=", "value": "a"}, KINDS), "not supported")
    expect_error(lambda: sb.build_condition({"column": "y", "op": "in", "value": " , "}, KINDS), "comma-separated")
    expect_error(lambda: sb.build_condition({"column": "y", "op": "in", "value": ",".join(["1"] * 501)}, KINDS), "comma-separated")


def test_null_operators_ignore_value_and_clean_filters():
    assert sb.build_condition({"column": "y", "op": "is not null", "value": "ignored"}, KINDS) == "`y` IS NOT NULL"
    rows = [{"column": None, "op": "=", "value": ""}, {"column": "y", "op": ">", "value": "1"}, {"column": "", "op": None, "value": None}]
    assert sb.clean_filters(rows) == [rows[1]]


# ---------------------------------------------------------------------------------------------------
# Joins. The generated SQL is executed in SQLite (it accepts backtick-quoted names), so these tests check real rows.
# ---------------------------------------------------------------------------------------------------
import sqlite3

LEFT_COLS = [("cell_id", "bigint"), ("chem", "string"), ("life", "double")]
RIGHT_COLS = [("cell_id", "bigint"), ("chem", "string"), ("temp", "double")]
PAIR = [{"left": "cell_id", "right": "cell_id"}]


def run_in_sqlite(plan, filters=(), select_columns=None):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE cells (cell_id INT, chem TEXT, life REAL)")
    db.executemany("INSERT INTO cells VALUES (?,?,?)", [(1, "A", 100), (2, "B", 200), (3, "A", 300)])
    db.execute("CREATE TABLE cycles (cell_id INT, chem TEXT, temp REAL)")
    db.executemany("INSERT INTO cycles VALUES (?,?,?)", [(1, "x", 20), (1, "x", 30), (2, "y", 25), (9, "z", 99)])
    kinds = {c["name"]: column_kind_of(c) for c in plan.columns}
    cursor = db.execute(sb.build_sql(plan.source_q, list(filters), kinds, select_columns))
    return [d[0] for d in cursor.description], cursor.fetchall()


def column_kind_of(column):
    return sb.column_kind(column["type"])


def make_plan(join_type="INNER", pairs=PAIR, left_cols=LEFT_COLS, right_cols=RIGHT_COLS):
    return sb.plan_join(sb.quote_ident("cells"), sb.quote_ident("cycles"), left_cols, right_cols, pairs, join_type)


def test_inner_join_rows_and_unique_columns():
    names, rows = run_in_sqlite(make_plan("INNER"))
    assert names == ["cell_id", "chem", "life", "temp"]                        # second cell_id and second chem are both left out
    assert len({n.lower() for n in names}) == len(names)
    # chem comes from the FIRST table (A, B), not the second table (x, y); cells 3 and 9 have no match
    assert sorted(rows) == [(1, "A", 100, 20), (1, "A", 100, 30), (2, "B", 200, 25)]


def test_left_join_keeps_unmatched_first_table_rows():
    names, rows = run_in_sqlite(make_plan("LEFT"))
    assert len(rows) == 4 and (3, "A", 300, None) in rows


def test_filters_work_on_joined_columns():
    plan = make_plan("INNER")
    kinds = {c["name"]: sb.column_kind(c["type"]) for c in plan.columns}
    names, rows = run_in_sqlite(plan, [{"column": "temp", "op": ">", "value": "22"}, {"column": "chem", "op": "in", "value": "A, B"}])
    assert sorted(r[3] for r in rows) == [25, 30]
    names, rows = run_in_sqlite(plan, [{"column": "chem", "op": "=", "value": "x"}])          # 'x' only exists in the second table's chem
    assert rows == []
    assert [c["name"] for c in plan.columns] == names                          # plan and real result agree on the columns


def test_plan_columns_and_notes():
    plan = make_plan()
    assert [(c["name"], c["origin"]) for c in plan.columns] == [("cell_id", "first"), ("chem", "first"), ("life", "first"), ("temp", "second")]
    assert any("only the first table's copy" in n and "'chem'" in n for n in plan.notes) and any("left out" in n for n in plan.notes)
    assert plan.source_q.startswith("(\nSELECT l.`cell_id`") and plan.source_q.endswith(") AS joined")


def test_different_key_names_and_multiple_keys():
    right = [("cid", "bigint"), ("chem", "string"), ("temp", "double")]
    plan = sb.plan_join("`cells`", "`cycles`", LEFT_COLS, right, [{"left": "cell_id", "right": "cid"}, {"left": "chem", "right": "chem"}], "INNER")
    assert [c["name"] for c in plan.columns] == ["cell_id", "chem", "life", "temp"]          # cid and chem(second) are both keys: dropped
    assert plan.source_q.count("ON l.`cell_id` = r.`cid`\n   AND l.`chem` = r.`chem`") == 1


def test_shared_columns_use_the_first_tables_copy_case_insensitively():
    left = [("id", "int"), ("Temp", "double"), ("life", "double")]
    right = [("id", "int"), ("TEMP", "double"), ("temp", "double"), ("extra", "double")]
    plan = sb.plan_join("`l`", "`r`", left, right, [{"left": "id", "right": "id"}], "INNER")
    names = [c["name"] for c in plan.columns]
    assert names == ["id", "Temp", "life", "extra"], names                                   # TEMP and temp match Temp -> left out
    assert len({n.lower() for n in names}) == len(names)
    assert "TEMP" in [n for n in plan.notes if "both tables" in n][0] and "'temp'" in [n for n in plan.notes if "both tables" in n][0]
    assert "r.`TEMP`" not in plan.source_q and "r.`extra`" in plan.source_q and "_right" not in plan.source_q


def test_no_note_when_nothing_is_shared():
    plan = sb.plan_join("`l`", "`r`", [("id", "int"), ("a", "int")], [("id", "int"), ("b", "int")], [{"left": "id", "right": "id"}], "LEFT")
    assert [c["name"] for c in plan.columns] == ["id", "a", "b"] and not any("both tables" in n for n in plan.notes)


def test_join_validation_errors():
    expect_error(lambda: make_plan(pairs=[]), "at least one pair")
    expect_error(lambda: make_plan(pairs=[{"left": "nope", "right": "cell_id"}]), "first table")
    expect_error(lambda: make_plan(pairs=[{"left": "cell_id", "right": "nope"}]), "second table")
    expect_error(lambda: make_plan(pairs=PAIR + PAIR), "twice")
    expect_error(lambda: make_plan(join_type="CROSS"), "Unsupported join type")
    expect_error(lambda: make_plan(right_cols=[("cell_id", "bigint")]), "adds no new columns")
    expect_error(lambda: make_plan(right_cols=[("cell_id", "bigint"), ("CHEM", "string")]), "adds no new columns")   # only a shared column


def test_clean_key_pairs():
    rows = [{"first_table_column": None, "second_table_column": None},
            {"first_table_column": "a", "second_table_column": "b"},
            {"first_table_column": float("nan"), "second_table_column": ""}]
    assert sb.clean_key_pairs(rows) == [{"left": "a", "right": "b"}]
    expect_error(lambda: sb.clean_key_pairs([{"first_table_column": "a", "second_table_column": None}]), "BOTH tables")


def test_join_labels_map_to_sql_keywords():
    assert sorted(sb.JOIN_TYPES.values()) == ["INNER", "LEFT"]


# ---------------------------------------------------------------------------------------------------
# 'contains': case-insensitive, and the user's own % _ \ are matched literally (they are not wildcards)
# ---------------------------------------------------------------------------------------------------
import re


def test_contains_sql_shape():
    cond = sb.build_condition({"column": "experiment", "op": "contains", "value": "Drone Design"}, {"experiment": "text"})
    assert cond == "lower(`experiment`) LIKE '%drone design%'"                 # both sides lower-cased
    assert sb.build_condition({"column": "experiment", "op": "contains", "value": "  Drone  "}, {"experiment": "text"}) == "lower(`experiment`) LIKE '%drone%'"
    assert "contains" in sb.OPERATORS


def test_contains_only_on_text_columns():
    for kind in ("numeric", "boolean", "datetime", "unsupported"):
        expect_error(lambda: sb.build_condition({"column": "c", "op": "contains", "value": "x"}, {"c": kind}), "not supported" if kind == "unsupported" else "only works on text")
    expect_error(lambda: sb.build_condition({"column": "c", "op": "contains", "value": "  "}, {"c": "text"}), "needs a value")


def spark_string_literal_value(literal):
    """What Spark's parser makes of a quoted SQL string literal (only the escapes our code can produce: '' and \\)."""
    assert literal[0] == "'" and literal[-1] == "'"
    inner, out, i = literal[1:-1], [], 0
    while i < len(inner):
        if inner[i] == "'" and inner[i + 1:i + 2] == "'":
            out.append("'"); i += 2
        elif inner[i] == "\\":
            assert inner[i + 1:i + 2] == "\\", "a lone backslash would be an escape sequence in Spark"
            out.append("\\"); i += 2
        else:
            out.append(inner[i]); i += 1
    return "".join(out)


def like_matches(pattern, text):
    """Spark LIKE with the default escape character (backslash): % = any run, _ = any one char, \\x = literal x."""
    regex, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            regex.append(re.escape(pattern[i + 1])); i += 2
        elif c == "%":
            regex.append(".*"); i += 1
        elif c == "_":
            regex.append("."); i += 1
        else:
            regex.append(re.escape(c)); i += 1
    return re.fullmatch("".join(regex), text, re.S) is not None


def sql_contains(needle, haystack):
    cond = sb.build_condition({"column": "c", "op": "contains", "value": needle}, {"c": "text"})
    literal = re.fullmatch(r"lower\(`c`\) LIKE ('.*')", cond, re.S).group(1)
    return like_matches(spark_string_literal_value(literal), haystack.lower())      # the engine lower-cases the column


def test_contains_means_literal_substring_ignoring_case():
    needles = ["drone design", "Drone Design", "100%", "a_b", "a\\b", "it's", "50% off_", "%", "_", "\\", "''", "x\\%y", "a.b*c", "[x]"]
    for needle in needles:
        haystacks = [f"pre {needle} post", needle, needle.upper(), needle.title(), "unrelated text", "1000", "aXb", "a b", "ab", "50x offy", "x\\y", "a.b*c!", "axb", "x", ""]
        for hay in haystacks:
            expected = needle.strip().lower() in hay.lower()
            assert sql_contains(needle, hay) == expected, f"needle={needle!r} haystack={hay!r}: SQL says {not expected}"


def test_contains_wildcard_characters_are_not_wildcards():
    assert sql_contains("100%", "100% sure") and not sql_contains("100%", "1000")          # % is not "anything"
    assert sql_contains("a_b", "xa_by") and not sql_contains("a_b", "aXb")                 # _ is not "any one char"


def test_contains_runs_in_sqlite_and_ignores_case():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE t (experiment TEXT, y REAL)")
    db.executemany("INSERT INTO t VALUES (?,?)", [("Drone Design v1", 1), ("DRONE DESIGN v2", 2), ("drone-design", 3), ("Fixed wing", 4), (None, 5)])
    sql = sb.build_sql("`t`", [{"column": "experiment", "op": "contains", "value": "drone design"}], {"experiment": "text", "y": "numeric"}, ["y"])
    assert sorted(r[0] for r in db.execute(sql)) == [1, 2]                               # any case; NULL and 'drone-design' excluded


# ---------------------------------------------------------------------------------------------------
# Choosing the columns to return
# ---------------------------------------------------------------------------------------------------
def test_select_columns_sql_and_dedupe():
    sql = sb.build_sql(T, [{"column": "y", "op": ">", "value": "1"}], KINDS, ["y", "name", "y"])
    assert sql == "SELECT `y`,\n       `name`\nFROM `main`.`sandbox`.`batteries`\nWHERE (`y` > 1)"
    assert sb.build_sql(T, [], KINDS, ["we`ird col"]) == "SELECT `we``ird col`\nFROM `main`.`sandbox`.`batteries`"
    assert sb.build_sql(T, [], KINDS, None) == "SELECT * FROM `main`.`sandbox`.`batteries`"             # None = all columns, as before


def test_select_columns_validation():
    expect_error(lambda: sb.build_sql(T, [], KINDS, ["nope"]), "Unknown column")
    expect_error(lambda: sb.build_sql(T, [], KINDS, []), "at least one column")


def test_filter_can_use_a_column_that_is_not_selected_and_joins_work_too():
    names, rows = run_in_sqlite(make_plan("INNER"), [{"column": "temp", "op": ">", "value": "22"}], ["life", "chem"])
    assert names == ["life", "chem"] and sorted(rows) == [(100, "A"), (200, "B")]          # temp is filtered on but not returned


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)