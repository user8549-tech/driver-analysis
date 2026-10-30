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
    assert sb.column_kind("string") == "text" and sb.column_kind("timestamp") == "text" and sb.column_kind("varchar(20)") == "text"
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)