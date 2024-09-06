from __future__ import annotations

import logging
import typing as t

from sqlglot import exp
from sqlglot import transforms
from sqlglot.dialects.dialect import (
    rename_func,
    if_sql,
)
from sqlglot.dialects.spark import Spark
from sqlglot.tokens import Tokenizer, TokenType

logger = logging.getLogger("sqlglot")

try:
    import local_clickzetta_settings
except ImportError as e:
    logger.warning(f'Failed to import local_clickzetta_settings, reason: {e}')
    pass


def _transform_create(expression: exp.Expression) -> exp.Expression:
    """Remove index column constraints.
    Remove unique column constraint (due to not buggy input)."""
    schema = expression.this
    if isinstance(expression, exp.Create) and isinstance(schema, exp.Schema):
        to_remove = []
        for e in schema.expressions:
            if isinstance(e, exp.IndexColumnConstraint) or \
                    isinstance(e, exp.UniqueColumnConstraint):
                to_remove.append(e)
        for e in to_remove:
            schema.expressions.remove(e)
    return expression

def _groupconcat_to_wmconcat(self: ClickZetta.Generator, expression: exp.GroupConcat) -> str:
    this = self.sql(expression, "this")
    sep = expression.args.get('separator')
    if not sep:
        sep = exp.Literal.string(',')
    return f"WM_CONCAT({sep}, {self.sql(this)})"

def _anonymous_func(self: ClickZetta.Generator, expression: exp.Anonymous) -> str:
    if expression.this.upper() == 'GETDATE':
        return f"CURRENT_TIMESTAMP()"
    elif expression.this.upper() == 'LAST_DAY_OF_MONTH':
        return f"LAST_DAY({self.sql(expression.expressions[0])})"
    elif expression.this.upper() == 'TO_ISO8601':
        return f"DATE_FORMAT({self.sql(expression.expressions[0])}, 'yyyy-MM-dd\\\'T\\\'hh:mm:ss.SSSxxx')"
    elif expression.this.upper() == 'MAP_AGG':
        return f"MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT({self.expressions(expression)})))"
    elif expression.this.upper() == 'JSON_ARRAY_GET':
        if len(expression.expressions) != 2:
            raise ValueError(f'JSON_ARRAY_GET needs 2 args, got {len(expression.expressions)}')
        arg1 = self.sql(expression.expressions[0])
        arg2 = self.sql(expression.expressions[1])
        return f"IF(TYPEOF({arg1}) == 'json', ({arg1}::JSON)[{arg2}], PARSE_JSON({arg1}::STRING)[{arg2}])"
    elif expression.this.upper() == 'PARSE_DATETIME':
        return f"TO_TIMESTAMP({self.sql(expression.expressions[0])}, {self.sql(expression.expressions[1])})"
    elif expression.this.upper() == 'FROM_ISO8601_TIMESTAMP':
        return f"CAST({self.sql(expression.expressions[0])} AS TIMESTAMP)"
    elif expression.this.upper() == 'DOW':
        # dow in presto is an alias of day_of_week, which is equivalent to dayofweek_iso
        # https://prestodb.io/docs/current/functions/datetime.html#day_of_week-x-bigint
        # https://doc.clickzetta.com/en-US/sql_functions/scalar_functions/datetime_functions/dayofweek_iso
        return f"DAYOFWEEK_ISO({self.sql(expression.expressions[0])})"
    elif expression.this.upper() == 'DOY':
        return f"DAYOFYEAR({self.sql(expression.expressions[0])})"
    elif expression.this.upper() == 'YOW' or expression.this.upper() == 'YEAR_OF_WEEK':
        return f"YEAROFWEEK({self.sql(expression.expressions[0])})"
    elif expression.this.upper() == 'GROUPING':
        return f"GROUPING_ID({self.expressions(expression, flat=True)})"
    elif expression.this.upper() == 'MURMUR_HASH3_32':
        return f"MURMURHASH3_32({self.sql(expression.expressions[0])})"

    # return as it is
    args = ", ".join(self.sql(e) for e in expression.expressions)
    return f"{expression.this}({args})"

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
    if isinstance(expression.expressions, list) and len(expression.expressions) == 1 and isinstance(
            expression.expressions[0], exp.Array):
        array = expression.expressions[0].expressions
        alias = expression.args.get('alias')
        ret = exp.Values(expressions=array, alias=alias)
        return self.sql(ret)
    elif len(expression.expressions) == 1:
        ret = f"EXPLODE({self.sql(expression.expressions[0])})"
        alias = expression.args.get('alias')
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
    unit = expression.args.get('unit')
    if not unit:
        unit = exp.Literal.string("DAY")
    if isinstance(unit, exp.Var):
        unit_str = f"'{self.sql(unit)}'"
    else:
        unit_str = self.sql(unit)
    return f"TIMESTAMP_OR_DATE_ADD({unit_str}, {self.sql(expression.expression)}, {self.sql(expression.this)})"


def _transform_group_sql(expression: exp.Expression) -> exp.Expression:
    # Handle CUBE
    cube = expression.args.get("cube", [])
    group_exprs = expression.expressions
    # If the cube list is only "true" not Column expressions, then convert to Column expressions
    if cube and isinstance(cube[0], exp.Cube):
        exprs = cube[0].expressions
        if not exprs:
            cube[0].set("expressions", group_exprs)
            return exp.Group(
                cube=cube,
                expressions=[]
            )

    # Handle ROLLUP
    rollup = expression.args.get("rollup", [])
    if rollup and isinstance(rollup[0], exp.Rollup):
        exprs = rollup[0].expressions
        if not exprs:
            rollup[0].set("expressions", group_exprs)
            return exp.Group(
                cube=rollup,
                expressions=[]
            )

    # Handle GROUPING SETS
    grouping = expression.args.get("grouping_sets", [])
    if grouping and isinstance(grouping[0], exp.GroupingSets):
        return exp.Group(
            grouping_sets=grouping,
            expressions=[]
        )

    # If no special clauses, return the original expression
    return expression


def _json_extract(name: str, self: ClickZetta.Generator, e: exp.JSONExtract) -> str:
    path = e.expression
    if isinstance(path, exp.Literal) and isinstance(path.this, str):
        # If the literal starts with $., use JSON_EXTRACT directly
        if path.this.startswith('$.'):
            return self.func(name, e.this, f"'{path.this}'", *e.expressions)
        return f"{e.this}['{e.expression.this}']"
    else:
        return self.func(name, e.this, self.sql(path), *e.expressions)


def _parse_json(self, e):
    if isinstance(e.this, exp.Literal) and e.this.is_string:
        return f"JSON '{e.this.this}'"
    return self.func("PARSE_JSON", e.this)


class ClickZetta(Spark):
    NULL_ORDERING = "nulls_are_small"
    LOG_BASE_FIRST = None

    class Tokenizer(Spark.Tokenizer):
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "CREATE USER": TokenType.COMMAND,
            "DROP USER": TokenType.COMMAND,
            "SHOW USER": TokenType.COMMAND,
            "REVOKE": TokenType.COMMAND,
        }

    class Parser(Spark.Parser):
        PROPERTY_PARSERS = {
            **Spark.Parser.PROPERTY_PARSERS,
            # ClickZetta has properties syntax similar to MySQL. e.g. PROPERTIES('key1'='value')
            "PROPERTIES": lambda self: self._parse_wrapped_properties(),
        }

    class Generator(Spark.Generator):
        RESERVED_KEYWORDS = {'all', 'user', 'to', 'check', 'order'}
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
        }

        PROPERTIES_LOCATION = {
            **Spark.Generator.PROPERTIES_LOCATION,
            exp.PrimaryKey: exp.Properties.Location.POST_NAME,
            exp.EngineProperty: exp.Properties.Location.POST_SCHEMA,
        }

        TRANSFORMS = {
            **Spark.Generator.TRANSFORMS,
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_qualify,
                    transforms.eliminate_distinct_on,
                    unnest_to_explode,
                ]
            ),
            # in MaxCompute, datetime(col) is an alias of cast(col as datetime)
            exp.Datetime: rename_func("TO_TIMESTAMP"),
            exp.DefaultColumnConstraint: lambda self, e: '',
            exp.DuplicateKeyProperty: lambda self, e: '',
            exp.OnUpdateColumnConstraint: lambda self, e: '',
            exp.AutoIncrementColumnConstraint: lambda self, e: '',
            exp.CollateColumnConstraint: lambda self, e: '',
            exp.CharacterSetColumnConstraint: lambda self, e: '',
            exp.Create: transforms.preprocess([_transform_create]),
            exp.GroupConcat: _groupconcat_to_wmconcat,
            exp.CurrentTime: lambda self, e: "DATE_FORMAT(NOW(),'HH:mm:ss')",
            exp.Anonymous: _anonymous_func,
            exp.AtTimeZone: lambda self, e: self.func(
                "CONVERT_TIMEZONE", e.args.get("zone"), e.this
            ),
            exp.UnixToTime: lambda self, e: self.func(
                "CONVERT_TIMEZONE", "'UTC+0'", e.this
            ),
            exp.EngineProperty: lambda self, e: '',
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
            exp.Chr: lambda self, e: f"CHAR({self.sql(exp.cast(e.this, exp.DataType.Type.INT))})",
            exp.ArrayAgg: rename_func("COLLECT_LIST"),
            exp.FromISO8601Timestamp: lambda self, e: f"{self.sql(exp.cast(e.this, exp.DataType.Type.TIMESTAMP))}",
        }

        def datatype_sql(self, expression: exp.DataType) -> str:
            """Remove unsupported type params from int types: eg. int(10) -> int
            Remove type param from enum series since it will be mapped as STRING."""
            type_value = expression.this
            type_sql = (
                self.TYPE_MAPPING.get(type_value, type_value.value)
                if isinstance(type_value, exp.DataType.Type)
                else type_value
            )
            if type_value in exp.DataType.INTEGER_TYPES or \
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
            }:
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

        def preprocess(self, expression: exp.Expression) -> exp.Expression:
            """Apply generic preprocessing transformations to a given expression."""

            # do not move ctes to top levels

            if self.ENSURE_BOOLS:
                from sqlglot.transforms import ensure_bools

                expression = ensure_bools(expression)

            return expression