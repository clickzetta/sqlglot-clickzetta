from __future__ import annotations

import logging
import typing as t

from sqlglot import exp
from sqlglot import transforms
from sqlglot.dialects.dialect import (
    rename_func,
    if_sql, unit_to_str,
    DATE_ADD_OR_SUB,
)
from sqlglot.dialects.hive import DATE_DELTA_INTERVAL
from sqlglot.dialects.spark import Spark
from sqlglot.dialects.mysql import MySQL
from sqlglot.tokens import Tokenizer, TokenType

logger = logging.getLogger("sqlglot")

try:
    from sqlglot import local_clickzetta_settings
except ImportError as e:
    logger.error(f"Failed to import local_clickzetta_settings, reason: {e}")


def _anonymous_agg_func(self: ClickZetta.Generator, expression: exp.AnonymousAggFunc) -> str:
    if expression.this.upper() == "UNIQEXACT":
        return self.sql(exp.Count(this=exp.Distinct(expressions=expression)))
    return self.func(expression.this, *expression.expressions)


# The current hive's _add_date_sql will lack parentheses, rewrite it
def _add_date_sql(self: ClickZetta.Generator, expression: DATE_ADD_OR_SUB) -> str:
    if isinstance(expression, exp.TsOrDsAdd) and not expression.unit:
        return self.func("DATE_ADD", expression.this, expression.expression)

    unit = expression.text("unit").upper()
    func, multiplier = DATE_DELTA_INTERVAL.get(unit, ("DATE_ADD", 1))

    if isinstance(expression, exp.DateSub):
        multiplier *= -1

    expr = expression.expression
    if expr.is_number:
        modified_increment = exp.Literal.number(expr.to_py() * multiplier)
    else:
        # We put parentheses on the expression, such as (1 + 1) * -1 translation
        modified_increment = expr if isinstance(expr, exp.Paren) else exp.Paren(this=expr)
        if multiplier != 1:
            modified_increment = exp.Mul(  # type: ignore
                this=modified_increment, expression=exp.Literal.number(multiplier)
            )

    return self.func(func, expression.this, modified_increment)


def _transform_create(expression: exp.Expression) -> exp.Expression:
    """Remove index column constraints.
    Remove unique column constraint (due to not buggy input)."""
    schema = expression.this
    if isinstance(expression, exp.Create) and isinstance(schema, exp.Schema):
        to_remove = []
        for e in schema.expressions:
            if isinstance(e, exp.IndexColumnConstraint) or isinstance(
                e, exp.UniqueColumnConstraint
            ):
                to_remove.append(e)
        for e in to_remove:
            schema.expressions.remove(e)
    return expression


def _string_agg_sql(self: ClickZetta.Generator, expression: exp.GroupConcat) -> str:
    separator = self.sql(expression, "separator")
    return f"""GROUP_CONCAT({self.sql(expression, "this")}{f' SEPARATOR {separator}' if separator else ''})"""


def _anonymous_func(self: ClickZetta.Generator, expression: exp.Anonymous) -> str:
    dialect = self.sql(expression.parent_select, "dialect")
    upper_name = expression.this.upper()
    if upper_name == "GETDATE":
        return "CURRENT_TIMESTAMP()"
    elif upper_name == "LAST_DAY_OF_MONTH":
        return f"LAST_DAY({self.sql(expression.expressions[0])})"
    elif upper_name == "TO_ISO8601":
        return f"DATE_FORMAT({self.sql(expression.expressions[0])}, 'yyyy-MM-dd\\'T\\'hh:mm:ss.SSSxxx')"
    elif upper_name == "MAP_AGG":
        return f"MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT({self.expressions(expression)})))"
    elif upper_name == "JSON_ARRAY_GET":
        if len(expression.expressions) != 2:
            raise ValueError(f"JSON_ARRAY_GET needs 2 args, got {len(expression.expressions)}")
        arg1 = self.sql(expression.expressions[0])
        arg2 = self.sql(expression.expressions[1])
        return f"IF(TYPEOF({arg1}) == 'json', ({arg1}::JSON)[{arg2}], PARSE_JSON({arg1}::STRING)[{arg2}])"
    elif upper_name == "PARSE_DATETIME":
        return f"TO_TIMESTAMP({self.sql(expression.expressions[0])}, {self.sql(expression.expressions[1])})"
    elif upper_name == "FROM_ISO8601_TIMESTAMP":
        return f"CAST({self.sql(expression.expressions[0])} AS TIMESTAMP)"
    elif upper_name == "DATE_FORMAT_MYSQL":
        if len(expression.expressions) >= 2:
            arg2 = self.sql(expression.expressions[1])
            if arg2 == "'yyyyMMdd'" or arg2 == "'yyyy-MM-dd'" or arg2 == "'yyyy-MM-dd HH:mm:ss'":
                return f"DATE_FORMAT({self.sql(expression.expressions[0])}, {self.sql(expression.expressions[1])})"
        return self.func(expression.this, *expression.expressions)
    elif upper_name == "UNIX_TIMESTAMP":
        if len(expression.expressions) >= 2:
            if dialect in ("DORIS", "STARROCKS", "MYSQL"):
                # The doris unix_timestamp is using mysql dateformat
                # https://doris.apache.org/zh-CN/docs/4.x/sql-manual/sql-functions/scalar-functions/date-time-functions/unix-timestamp/
                # First convert MySQL format to Python strftime format
                mysql_format = self.sql(expression.expressions[1], "this")
                from sqlglot.time import format_time
                python_format = format_time(mysql_format, MySQL.TIME_MAPPING, MySQL.TIME_TRIE)
                # Then convert Python strftime format to ClickZetta format
                if python_format:
                    clickzetta_format = format_time(python_format, self.dialect.INVERSE_TIME_MAPPING,
                                                    self.dialect.INVERSE_TIME_TRIE)
                    return f"UNIX_TIMESTAMP({self.sql(expression.expressions[0])}, '{clickzetta_format}')"
                else:
                    # If conversion fails, use the original format
                    return self.func(expression.this, *expression.expressions)
        return self.func(expression.this, *expression.expressions)
    elif upper_name == "DOW":
        # dow in presto is an alias of day_of_week, which is equivalent to dayofweek_iso
        # https://prestodb.io/docs/current/functions/datetime.html#day_of_week-x-bigint
        # https://doc.clickzetta.com/en-US/sql_functions/scalar_functions/datetime_functions/dayofweek_iso
        return f"DAYOFWEEK_ISO({self.sql(expression.expressions[0])})"
    elif upper_name == "DOY":
        return f"DAYOFYEAR({self.sql(expression.expressions[0])})"
    elif upper_name == "YOW" or upper_name == "YEAR_OF_WEEK":
        return f"YEAROFWEEK({self.sql(expression.expressions[0])})"
    elif upper_name == "GROUPING":
        return f"GROUPING_ID({self.expressions(expression, flat=True)})"
    elif upper_name == "MURMUR_HASH3_32":
        return f"MURMURHASH3_32({self.sql(expression.expressions[0])})"
    elif upper_name == "GET_JSON_OBJECT":
        return _build_parse_json_sql(
            upper_name, self, expression.expressions[0], expression.expressions[1], dialect=dialect
        )
    elif (
        "VISITPARAMEXTRACT" in upper_name
        or "JSONEXTRACT" in upper_name
        or "SIMPLEJSONEXTRACT" in upper_name
    ):
        # VISITPARAMEXTRACT is alias of SIMPLEJSONEXTRACT in ClickHouse
        # The JSONEXTRACT parameters indices_or_keys currently not supported
        # https://clickhouse.com/docs/en/sql-reference/functions/json-functions#simplejsonextractfloat
        upper_name_subfix = upper_name.replace("VISITPARAMEXTRACT", "").replace("JSONEXTRACT", "")
        func_name = "JSON_EXTRACT"
        if upper_name_subfix == "BOOL":
            func_name = "JSON_EXTRACT_BOOLEAN"
        elif upper_name_subfix == "INT":
            func_name = "JSON_EXTRACT_BIGINT"
        elif upper_name_subfix == "FLOAT":
            func_name = "JSON_EXTRACT_DOUBLE"
        elif upper_name_subfix == "STRING":
            func_name = "JSON_EXTRACT_STRING"
        exprs = expression.expressions
        extract_sql = _build_parse_json_sql(func_name, self, exprs[0], exprs[1], dialect=dialect)
        if upper_name_subfix == "ARRAYRAW":
            extract_sql = f"{extract_sql}::ARRAY<JSON>"
        return extract_sql
    elif upper_name == "REPLACEALL":
        return self.func(
            "REPLACE",
            expression.expressions[0],
            expression.expressions[1],
            expression.expressions[2],
        )
    elif upper_name == "TOSTARTOFDAY":
        return f"DATE_TRUNC('DAY', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFHOUR":
        return f"DATE_TRUNC('HOUR', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFMINUTE":
        return f"DATE_TRUNC('MINUTE', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFMONTH":
        return f"DATE_TRUNC('MONTH', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFQUARTER":
        return f"DATE_TRUNC('QUARTER', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFSECOND":
        return f"DATE_TRUNC('SECOND', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFWEEK":
        return f"DATE_TRUNC('WEEK', {self.sql(expression.expressions[0])})"
    elif upper_name == "TOSTARTOFYEAR":
        return f"DATE_TRUNC('YEAR', {self.sql(expression.expressions[0])})"
    # return as it is
    return self.func(expression.this, *expression.expressions)


def nullif_to_if(self: ClickZetta.Generator, expression: exp.Nullif):
    cond = exp.EQ(this=expression.this, expression=expression.expression)
    ret = exp.If(this=cond, true=exp.Null(), false=expression.this)
    return self.sql(ret)


def unnest_to_explode(
    expression: exp.Expression,
    unnest_using_arrays_zip: bool = True,
    generator: t.Optional[ClickZetta.Generator] = None,
) -> exp.Expression:
    """Convert cross join unnest into lateral view explode."""

    def _unnest_zip_exprs(
        u: exp.Unnest, unnest_exprs: t.List[exp.Expression], has_multi_expr: bool
    ) -> t.List[exp.Expression]:
        if has_multi_expr:
            if not unnest_using_arrays_zip:
                if generator:
                    generator.unsupported(
                        f"Multiple expressions in UNNEST are not supported in "
                        f"{generator.dialect.__module__.split('.')[-1].upper()}"
                    )
            else:
                # Use INLINE(ARRAYS_ZIP(...)) for multiple expressions
                zip_exprs: t.List[exp.Expression] = [
                    exp.Anonymous(this="ARRAYS_ZIP", expressions=unnest_exprs)
                ]
                u.set("expressions", zip_exprs)
                return zip_exprs
        return unnest_exprs

    def _udtf_type(u: exp.Unnest, has_multi_expr: bool) -> t.Type[exp.Func]:
        if u.args.get("offset"):
            return exp.Posexplode
        return exp.Inline if has_multi_expr else exp.Explode

    if isinstance(expression, exp.Select):
        for join in expression.args.get("joins") or []:
            join_expr = join.this

            is_lateral = isinstance(join_expr, exp.Lateral)

            unnest = join_expr.this if is_lateral else join_expr

            if isinstance(unnest, exp.Unnest):
                if is_lateral:
                    alias = join_expr.args.get("alias")
                else:
                    alias = unnest.args.get("alias")
                exprs = unnest.expressions
                # The number of unnest.expressions will be changed by _unnest_zip_exprs, we need to record it here
                has_multi_expr = len(exprs) > 1
                exprs = _unnest_zip_exprs(unnest, exprs, has_multi_expr)

                expression.args["joins"].remove(join)

                alias_cols = alias.columns if alias else []
                for e, column in zip(exprs, alias_cols):
                    expression.append(
                        "laterals",
                        exp.Lateral(
                            this=_udtf_type(unnest, has_multi_expr)(this=e),
                            view=True,
                            alias=exp.TableAlias(
                                this=alias.this,  # type: ignore
                                columns=alias_cols if unnest_using_arrays_zip else [column],  # type: ignore
                            ),
                        ),
                    )

    return expression


def unnest_to_values(self: ClickZetta.Generator, expression: exp.Unnest):
    # If the array contains only tuples, the array is treated as a single map column.
    # https://prestodb.io/docs/current/sql/select.html#unnest
    exprs = expression.expressions
    if (
        isinstance(exprs, list)
        and len(exprs) == 1
        and isinstance(exprs[0], exp.Array)
        and all(isinstance(e, exp.Tuple) for e in exprs[0].expressions)
    ):
        array = exprs[0].expressions
        alias = expression.args.get("alias")
        ret = exp.Values(expressions=array, alias=alias)
        return self.sql(ret)
    elif len(expression.expressions) == 1:
        ret = f"EXPLODE({self.sql(expression.expressions[0])})"
        alias = expression.args.get("alias")
        if alias:
            ret = f"{ret} AS {self.tablealias_sql(expression.args.get('alias'))}"
        return ret
    # Not set dialect, will call unnest_sql in generator.Generator to ensure that it will not be affected
    # by upstream changes
    return expression.sql()


def date_add_sql(self: ClickZetta.Generator, expression: exp.DateAdd) -> str:
    """
    Convert date_add to TIMESTAMP_OR_DATE_ADD.

    Note the currently difference between `exp.DateAdd` and `exp.TimestampAdd` and `exp.Anonymous(date_add)`
    | dialect | sql example | expression | return type |
    |---------|--------------|------------|--------------|
    | starrocks | select date_add('2010-11-30 23:59:59', INTERVAL 2 DAY) | exp.DateAdd | timestamp or date |
    | presto | select date_add('2010-11-30 23:59:59', INTERVAL 2 DAY) | exp.DateAdd | timestamp or date |
    | spark | select date_add('2024-01-01', 1) | exp.Anonymous | date |
    | spark | select dateadd(DAY, 1, '2024-01-01') | exp.TimestampAdd | timestamp |
    """
    # https://prestodb.io/docs/current/functions/datetime.html#date_add
    unit = expression.args.get("unit")
    if not unit:
        unit = exp.Literal.string("DAY")
    if isinstance(unit, exp.Var):
        unit_str = f"'{self.sql(unit)}'"
    else:
        unit_str = self.sql(unit)
    # canonicalize INTERVAL values to number literals
    interval = expression.expression
    if interval and interval.is_string:
        interval = exp.Literal.number(interval.this)
    return f"TIMESTAMP_OR_DATE_ADD({unit_str}, {self.sql(interval)}, {self.sql(expression.this)})"


def _transform_group_sql(expression: exp.Expression) -> exp.Expression:
    # Handle CUBE
    cube = expression.args.get("cube", [])
    group_exprs = expression.expressions
    # If the cube list is only "true" not Column expressions, then convert to Column expressions
    if cube and isinstance(cube[0], exp.Cube):
        exprs = cube[0].expressions
        if not exprs:
            cube[0].set("expressions", group_exprs)
            return exp.Group(cube=cube, expressions=[])

    # Handle ROLLUP
    rollup = expression.args.get("rollup", [])
    if rollup and isinstance(rollup[0], exp.Rollup):
        exprs = rollup[0].expressions
        if not exprs:
            rollup[0].set("expressions", group_exprs)
            return exp.Group(cube=rollup, expressions=[])

    # Handle GROUPING SETS
    grouping = expression.args.get("grouping_sets", [])
    if grouping and isinstance(grouping[0], exp.GroupingSets):
        return exp.Group(grouping_sets=grouping, expressions=[])

    # If no special clauses, return the original expression
    return expression


def _json_extract(
    name: str, self: ClickZetta.Generator, expression: exp.JSONExtract | exp.JSONExtractScalar
) -> str:
    return _build_parse_json_sql(
        name, self, expression.this, expression.expression, expression.expressions
    )


def _build_parse_json_sql(
    name: str,
    self: ClickZetta.Generator,
    json: exp.Expression,
    path: exp.Expression,
    exprs: t.List[exp.Expression] = (),
    dialect: str = "",
) -> str:
    if not dialect:
        dialect = self.sql(path.parent_select, "dialect")
    # If it is not a Literal type but a JsonPath, the $ prefix will be automatically added during translation
    if isinstance(path, exp.Literal) and isinstance(path.this, str):
        path_str = path.this
        # The key in Clickhouse's jsonpath contains a dot, not a hierarchical relationship, but a part of the key.
        # Other dialects do not need special processing
        if path_str.find(".") != -1 and dialect == "CLICKHOUSE":
            path_str = rf"'$[\'{path_str}\']'"
        # If the literal starts with $., use JSON_EXTRACT directly
        elif path_str.startswith("$."):
            path_str = f"'{path_str}'"
        else:
            path_str = f"'$.{path_str}'"
    else:
        # If it is not a Literal type but a JsonPath, the $ prefix will be automatically added during translation
        path_str = self.sql(path)

    if name.upper() != "GET_JSON_OBJECT" and isinstance(json, exp.Literal) and json.is_string:
        return self.func(name, exp.ParseJSON(this=json), path_str, *exprs)
    return self.func(name, json, path_str, *exprs)


def _parse_json(self, e: exp.ParseJSON) -> str:
    if isinstance(e.this, exp.Literal) and e.this.is_string:
        return f"JSON '{e.this.this}'"
    return self.func("PARSE_JSON", e.this)


class ClickZetta(Spark):
    NULL_ORDERING = "nulls_are_small"
    LOG_BASE_FIRST = None
    # https://github.com/tobymao/sqlglot/issues/4013
    STRICT_JSON_PATH_SYNTAX = False

    class Tokenizer(Spark.Tokenizer):
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "CREATE USER": TokenType.COMMAND,
            "DROP USER": TokenType.COMMAND,
            "SEPARATOR": TokenType.SEPARATOR,
            "SHOW USER": TokenType.COMMAND,
            "REVOKE": TokenType.COMMAND,
        }

    class Parser(Spark.Parser):
        FUNCTION_PARSERS = {
            **Spark.Parser.FUNCTION_PARSERS,
            "GROUP_CONCAT": lambda self: MySQL.Parser._parse_group_concat(self),
        }

        PROPERTY_PARSERS = {
            **Spark.Parser.PROPERTY_PARSERS,
            # ClickZetta has properties syntax similar to MySQL. e.g. PROPERTIES('key1'='value')
            "PROPERTIES": lambda self: self._parse_wrapped_properties(),
        }

        def _parse_unique_key_property(self):
            """Parse UNIQUE KEY syntax as a property."""
            expressions = self._parse_wrapped_csv(self._parse_id_var, optional=False)
            return self.expression(exp.UniqueKeyProperty, expressions=expressions)

        def _parse_types(self, check_func=False, schema=False, allow_identifiers=True):
            """Override to handle BITMAP type."""
            # Check if current token is BITMAP
            if self._match_text_seq("BITMAP"):
                if hasattr(exp.DataType.Type, "BITMAP"):
                    return exp.DataType(this=exp.DataType.Type.BITMAP, nested=False)
                else:
                    # Map BITMAP to HLLSKETCH internally for consistency
                    return exp.DataType(this=exp.DataType.Type.HLLSKETCH, nested=False)
            return super()._parse_types(check_func, schema, allow_identifiers)

        def _parse_schema(self, this=None):
            """Override to handle UNIQUE as a schema-level expression."""
            schema = super()._parse_schema(this)

            # If schema has UniqueColumnConstraint, convert it to UniqueKeyProperty
            if schema and hasattr(schema, 'expressions'):
                # Make a copy of the list to avoid modification during iteration
                unique_constraints = [e for e in schema.expressions if isinstance(e, exp.UniqueColumnConstraint)]
                for uc in unique_constraints:
                    # The UniqueColumnConstraint wraps either a Schema or UniqueKeyProperty
                    if hasattr(uc, 'this'):
                        # Remove UniqueColumnConstraint
                        schema.expressions.remove(uc)
                        # Extract the UniqueKeyProperty from inside
                        if isinstance(uc.this, exp.Schema) and hasattr(uc.this, 'expressions'):
                            # UniqueColumnConstraint(Schema(expressions=[...]))
                            # We need to create UniqueKeyProperty from the schema's expressions
                            unique_key = self.expression(exp.UniqueKeyProperty, expressions=uc.this.expressions)
                            schema.expressions.append(unique_key)
                        elif isinstance(uc.this, exp.UniqueKeyProperty):
                            schema.expressions.append(uc.this)

            return schema

        def _to_prop_eq(self, expression: exp.Expression, index: int) -> exp.Expression:
            # ClickZetta does not support add alias for STRUCT function, so we need to directly return the expression
            # Otherwise, if is useful, we can use to `named_struct` in the future
            return expression

    class Generator(Spark.Generator):
        RESERVED_KEYWORDS = {
            "all",
            "user",
            "to",
            "check",
            "order",
            "current_timestamp",
            "current_date",
        }
        WITH_PROPERTIES_PREFIX = "PROPERTIES"

        TYPE_MAPPING = {
            **Spark.Generator.TYPE_MAPPING,
            exp.DataType.Type.MEDIUMTEXT: "STRING",
            exp.DataType.Type.LONGTEXT: "STRING",
            exp.DataType.Type.VARIANT: "STRING",
            exp.DataType.Type.ENUM: "STRING",
            exp.DataType.Type.ENUM16: "STRING",
            exp.DataType.Type.ENUM8: "STRING",
            # mysql unsigned types
            exp.DataType.Type.UINT: "INT",
            exp.DataType.Type.UTINYINT: "TINYINT",
            exp.DataType.Type.USMALLINT: "SMALLINT",
            exp.DataType.Type.UMEDIUMINT: "INT",
            exp.DataType.Type.UBIGINT: "BIGINT",
            exp.DataType.Type.UDECIMAL: "DECIMAL",
            # postgres serial types
            exp.DataType.Type.BIGSERIAL: "BIGINT",
            exp.DataType.Type.SERIAL: "INT",
            exp.DataType.Type.SMALLSERIAL: "SMALLINT",
            exp.DataType.Type.BIGDECIMAL: "DECIMAL",
            # starrocks decimal types
            exp.DataType.Type.DECIMAL32: "DECIMAL",
            exp.DataType.Type.DECIMAL64: "DECIMAL",
            exp.DataType.Type.DECIMAL128: "DECIMAL",
            exp.DataType.Type.IPV4: "BINARY",
            exp.DataType.Type.IPV6: "BINARY",
            # We map HLLSKETCH to BINARY for StarRocks/Doris compatibility
            exp.DataType.Type.HLLSKETCH: "BINARY",
            "BITMAP": "BITMAP",
            "HLL": "BINARY",
            "LARGEINT": "BIGINT",
            "QUANTILE_STATE": "BINARY",
            "AGG_STATE": "BINARY",
        }

        PROPERTIES_LOCATION = {
            **Spark.Generator.PROPERTIES_LOCATION,
            exp.PrimaryKey: exp.Properties.Location.POST_NAME,
            exp.EngineProperty: exp.Properties.Location.POST_SCHEMA,
        }

        # Add UniqueKeyProperty location if it exists
        if hasattr(exp, 'UniqueKeyProperty'):
            PROPERTIES_LOCATION[exp.UniqueKeyProperty] = exp.Properties.Location.POST_SCHEMA

        TRANSFORMS = {
            **Spark.Generator.TRANSFORMS,
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_qualify,
                    transforms.eliminate_distinct_on,
                    unnest_to_explode,
                ]
            ),
            exp.Anonymous: _anonymous_func,
            exp.AnonymousAggFunc: _anonymous_agg_func,
            exp.CastToStrType: lambda self, e: "CAST({} AS {})".format(
                self.sql(e, "this"), self.sql(e, "to").strip("'").upper()
            ),
            # in MaxCompute, datetime(col) is an alias of cast(col as datetime)
            exp.Datetime: rename_func("TO_TIMESTAMP"),
            exp.DateTrunc: lambda self, e: self.func("DATE_TRUNC", unit_to_str(e), e.this),
            exp.DateSub: lambda self, e: _add_date_sql(self, e),
            exp.DefaultColumnConstraint: lambda self, e: "",
            exp.DuplicateKeyProperty: lambda self, e: "",
            exp.OnUpdateColumnConstraint: lambda self, e: "",
            exp.AutoIncrementColumnConstraint: lambda *_: "IDENTITY(1)",
            exp.CollateColumnConstraint: lambda self, e: "",
            exp.CharacterSetColumnConstraint: lambda self, e: "",
            exp.Create: transforms.preprocess([_transform_create]),
            exp.GroupConcat: _string_agg_sql,
            exp.CurrentTime: lambda self, e: "DATE_FORMAT(NOW(),'HH:mm:ss')",
            exp.AtTimeZone: lambda self, e: self.func(
                "CONVERT_TIMEZONE", e.args.get("zone"), e.this
            ),
            exp.EngineProperty: lambda self, e: "",
            exp.Pow: rename_func("POW"),
            exp.ApproxQuantile: rename_func("APPROX_PERCENTILE"),
            exp.JSONFormat: rename_func("TO_JSON"),
            exp.ParseJSON: _parse_json,
            exp.Nullif: nullif_to_if,
            exp.If: if_sql(false_value=exp.Null()),
            exp.Unnest: unnest_to_values,
            exp.Try: lambda self, e: self.sql(e, "this"),
            exp.GenerateSeries: rename_func("SEQUENCE"),
            exp.DateAdd: date_add_sql,
            exp.DayOfWeekIso: lambda self, e: self.func("DAYOFWEEK_ISO", e.this),
            exp.Group: transforms.preprocess([_transform_group_sql]),
            exp.RegexpLike: rename_func("RLIKE"),
            exp.JSONExtract: lambda self, e: _json_extract("JSON_EXTRACT", self, e),
            exp.JSONExtractScalar: lambda self, e: _json_extract("JSON_EXTRACT", self, e),
            exp.Chr: lambda self, e: self.func(
                "CHAR", exp.cast(self.sql(*e.expressions), exp.DataType.Type.INT)
            ),
            exp.ArrayAgg: rename_func("COLLECT_LIST"),
            exp.FromISO8601Timestamp: lambda self,
            e: f"{self.sql(exp.cast(e.this, exp.DataType.Type.TIMESTAMP))}",
        }

        def datatype_sql(self, expression: exp.DataType) -> str:
            """Remove unsupported type params from int types: eg. int(10) -> int
            Remove type param from enum series since it will be mapped as STRING."""
            type_value = expression.this

            # Check if type_value is an enum or a string
            if isinstance(type_value, exp.DataType.Type):
                type_sql = self.TYPE_MAPPING.get(type_value, type_value.value)
            else:
                # For string types (like our custom Doris types), check TYPE_MAPPING by string value
                type_sql = self.TYPE_MAPPING.get(type_value, type_value)

            # Check for Doris-specific type strings that need special handling
            special_types = {"LARGEINT", "HLL", "QUANTILE_STATE", "AGG_STATE", "BITMAP"}

            if (type_value in exp.DataType.INTEGER_TYPES or
                (isinstance(type_value, str) and type_value in special_types) or
                type_value in {
                    exp.DataType.Type.UTINYINT,
                    exp.DataType.Type.USMALLINT,
                    exp.DataType.Type.UMEDIUMINT,
                    exp.DataType.Type.UINT,
                    exp.DataType.Type.UINT128,
                    exp.DataType.Type.UINT256,
                    exp.DataType.Type.ENUM,
                    exp.DataType.Type.FLOAT,
                    exp.DataType.Type.DOUBLE,
                }):
                return type_sql
            return super().datatype_sql(expression)

        def distributedbyproperty_sql(self, expression: exp.DistributedByProperty) -> str:
            expressions = self.expressions(expression, key="expressions", flat=True)
            order = self.expressions(expression, key="order", flat=True)
            order = f" SORTED BY {order}" if order else ""
            buckets = self.sql(expression, "buckets")
            if not buckets:
                self.unsupported(
                    "DistributedByHash without buckets, clickzetta requires a number of buckets"
                )
            return f"CLUSTERED BY ({expressions}){order} INTO {buckets} BUCKETS"

        def uniquekeyproperty_sql(self, expression: exp.Expression) -> str:
            """Generate SQL for UniqueKeyProperty."""
            if hasattr(exp, 'UniqueKeyProperty') and isinstance(expression, exp.UniqueKeyProperty):
                expressions = self.expressions(expression, key="expressions", flat=True)
                return f"UNIQUE ({expressions})"
            return ""

        def preprocess(self, expression: exp.Expression) -> exp.Expression:
            """Apply generic preprocessing transformations to a given expression."""

            # do not move ctes to top levels

            if self.ENSURE_BOOLS:
                from sqlglot.transforms import ensure_bools

                expression = ensure_bools(expression)

            return expression
