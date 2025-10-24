import typing as t

from sqlglot import exp
from sqlglot._typing import E
from sqlglot.dialects.dialect import unit_to_str
from sqlglot.dialects.athena import Athena
from sqlglot.dialects.clickhouse import ClickHouse
from sqlglot.dialects.doris import Doris
from sqlglot.dialects.mysql import MySQL
from sqlglot.dialects.postgres import Postgres
from sqlglot.dialects.presto import Presto
from sqlglot.dialects.redshift import Redshift
from sqlglot.dialects.starrocks import StarRocks
from sqlglot.dialects.trino import Trino
from sqlglot.helper import seq_get
from sqlglot.parser import Parser


# https://github.com/tobymao/sqlglot/issues/4345
# Wait for the issue to be resolved before removing this workaround
def _build_date_delta_with_interval(
        expression_class: t.Type[E],
) -> t.Callable[[t.List], t.Optional[E]]:
    def _builder(args: t.List) -> t.Optional[E]:
        if len(args) < 2:
            return None

        interval = args[1]

        expression = None
        if isinstance(interval, exp.Interval):
            expression = interval.this
        else:
            expression = interval
        return expression_class(this=args[0], expression=expression, unit=unit_to_str(interval))

    return _builder


# Note: This workaround only allows the syntax that has been adapted to the Source Dialect of Clickzetta to be hacked.
# Anything that can be solved through expression will not be allowed here.
for dialect in [MySQL, Presto, Trino, Athena, StarRocks, Doris]:
    dialect.Parser.FUNCTIONS["DATE_FORMAT"] = lambda args: exp.Anonymous(
        this="DATE_FORMAT_MYSQL", expressions=args
    )
    dialect.Parser.FUNCTIONS["AES_DECRYPT"] = lambda args: exp.Anonymous(
        this="AES_DECRYPT_MYSQL", expressions=args
    )
    dialect.Parser.FUNCTIONS["AES_ENCRYPT"] = lambda args: exp.Anonymous(
        this="AES_ENCRYPT_MYSQL", expressions=args
    )

ClickHouse.Parser.FUNCTIONS["FORMATDATETIME"] = lambda args: exp.Anonymous(
    this="DATE_FORMAT_MYSQL", expressions=args
)

MySQL.Parser.FUNCTIONS["DATE_ADD"] = StarRocks.Parser.FUNCTIONS["DATE_ADD"] = Doris.Parser.FUNCTIONS[
    "DATE_ADD"] = _build_date_delta_with_interval(exp.DateAdd)
MySQL.Parser.FUNCTIONS["DATE_SUB"] = StarRocks.Parser.FUNCTIONS["DATE_SUB"] = Doris.Parser.FUNCTIONS[
    "DATE_SUB"] = _build_date_delta_with_interval(exp.DateSub)

for dialect in [Postgres, Redshift]:
    dialect.Parser.FUNCTIONS["TO_CHAR"] = lambda args: exp.Anonymous(
        this="DATE_FORMAT_PG", expressions=args
    )

# Add ClickHouse functions in a workaround way, delete after sqlglot supports it
ClickHouse.Parser.FUNCTIONS["FROMUNIXTIMESTAMP64MILLI"] = lambda args: exp.UnixToTime(
    this=seq_get(args, 0),
    zone=seq_get(args, 1) if len(args) == 2 else None,
    scale=exp.UnixToTime.MILLIS,
)

# Clickhouse's JSONExtract* and visitParamExtract*  will be parsed as JSONExtractScalar, which we do not support,
# and different types need to be processed separately.
# Notice: This will cause JSONEXTRACT* -> JSON_EXTRACT_PATH_TEXT related cases to fail.
ClickHouse.Parser.FUNCTIONS["JSONEXTRACTSTRING"] = lambda args: exp.Anonymous(
    this="JSONEXTRACTSTRING", expressions=args
)
ClickHouse.Parser.FUNCTIONS["VISITPARAMEXTRACTSTRING"] = lambda args: exp.Anonymous(
    this="VISITPARAMEXTRACTSTRING", expressions=args
)
ClickHouse.Parser.FUNCTIONS["VISITPARAMEXTRACTRAW"] = lambda args: exp.Anonymous(
    this="GET_JSON_OBJECT", expressions=args
)
ClickHouse.Parser.FUNCTIONS["SIMPLEJSONEXTRACTRAW"] = lambda args: exp.Anonymous(
    this="GET_JSON_OBJECT", expressions=args
)
ClickHouse.Parser.FUNCTIONS["JSONEXTRACTRAW"] = lambda args: exp.Anonymous(
    this="GET_JSON_OBJECT", expressions=args
)

# ClickHouse's toDateTime(expr[, timezone]) parameter expr supports String, Int, Date or DateTime.
# To adapt to multiple types, we use the cast function for conversion.
# Notice: This will cause TODATETIME -> CAST related cases to fail.
ClickHouse.Parser.FUNCTIONS["TODATETIME"] = lambda args: exp.cast(
    seq_get(args, 0), exp.DataType.Type.DATETIME
)
ClickHouse.Parser.FUNCTIONS["TODATE"] = lambda args: exp.cast(
    seq_get(args, 0), exp.DataType.Type.DATE
)

_parse_select = getattr(Parser, "_parse_select")


def preprocess_parse_select(self, *args, **kwargs):
    expression = _parse_select(self, *args, **kwargs)
    if not expression:
        return expression
    # source dialect
    read_dialect = self.dialect.__module__.split(".")[-1].upper()
    expression.set("dialect", read_dialect)
    if read_dialect == "PRESTO":
        _normalize_tuple_comparisons(expression)
    return expression


setattr(Parser, "_parse_select", preprocess_parse_select)


# According to #4042 suggestion, we create a custom transformation to handle presto tuple comparisons
# https://github.com/tobymao/sqlglot/issues/4042
def _normalize_tuple_comparisons(expression: exp.Expression):
    for tup in expression.find_all(exp.Tuple):
        if not isinstance(tup.parent, exp.Binary) or not isinstance(tup.parent, exp.Predicate):
            continue
        binary = tup.parent
        left, right = binary.this, binary.expression
        if not isinstance(left, exp.Tuple) or not isinstance(right, exp.Tuple):
            continue
        left_exprs = left.expressions
        right_exprs = right.expressions
        for i, (left_expr, right_expr) in enumerate(zip(left_exprs, right_exprs), start=1):
            alias = f"col{i}"
            if not isinstance(left_expr, exp.Alias):
                left_exprs[i - 1] = exp.Alias(this=left_expr, alias=exp.to_identifier(alias))
            else:
                left_exprs[i - 1].set("alias", exp.to_identifier(alias))

            if not isinstance(right_expr, exp.Alias):
                right_exprs[i - 1] = exp.Alias(this=right_expr, alias=exp.to_identifier(alias))
            else:
                right_exprs[i - 1].set("alias", exp.to_identifier(alias))


# Add UniqueKeyProperty expression class if it doesn't exist
if not hasattr(exp, 'UniqueKeyProperty'):
    class UniqueKeyProperty(exp.Property):
        arg_types = {"expressions": True}


    # Register it in the exp module
    exp.UniqueKeyProperty = UniqueKeyProperty

# Monkey patch Doris and StarRocks parsers to support UNIQUE KEY and BITMAP
original_doris_parse_create = Doris.Parser._parse_create
original_starrocks_parse_create = StarRocks.Parser._parse_create


def _parse_unique_key(self):
    """Parse UNIQUE KEY syntax."""
    self._match_text_seq("KEY")
    expressions = self._parse_wrapped_csv(self._parse_id_var, optional=False)
    return self.expression(exp.UniqueKeyProperty, expressions=expressions)


def _patched_doris_parse_create(self):
    """Patched Doris _parse_create to support UNIQUE KEY and move it to schema."""
    create = original_doris_parse_create(self)

    # Move UniqueKey from properties to schema (similar to PrimaryKey handling in StarRocks)
    if isinstance(create, exp.Create) and isinstance(create.this, exp.Schema):
        props = create.args.get("properties")
        if props:
            unique_key = props.find(exp.UniqueKeyProperty)
            if unique_key:
                create.this.append("expressions", unique_key.pop())

    return create


def _patched_starrocks_parse_create(self):
    """Patched StarRocks _parse_create to support UNIQUE KEY and move it to schema."""
    create = original_starrocks_parse_create(self)

    # Move UniqueKey from properties to schema (similar to PrimaryKey handling)
    if isinstance(create, exp.Create) and isinstance(create.this, exp.Schema):
        props = create.args.get("properties")
        if props:
            unique_key = props.find(exp.UniqueKeyProperty)
            if unique_key:
                create.this.append("expressions", unique_key.pop())

    return create


# Apply monkey patches
Doris.Parser._parse_create = _patched_doris_parse_create
Doris.Parser._parse_unique = _parse_unique_key
StarRocks.Parser._parse_create = _patched_starrocks_parse_create
StarRocks.Parser._parse_unique = _parse_unique_key

# Add UNIQUE to PROPERTY_PARSERS for both dialects
for dialect in [Doris, StarRocks]:
    if hasattr(dialect.Parser, 'PROPERTY_PARSERS'):
        dialect.Parser.PROPERTY_PARSERS = {
            **dialect.Parser.PROPERTY_PARSERS,
            "UNIQUE": lambda self: self._parse_unique(),
        }

# Add BITMAP data type support
# We need to add BITMAP to the DataType.Type enum
# Since we can't easily extend AutoName enums at runtime, we'll use a workaround
# by treating BITMAP as a special identifier that gets mapped to a custom type

# Store the original _parse_types method for Doris and StarRocks
original_doris_parse_types = Doris.Parser._parse_types
original_starrocks_parse_types = StarRocks.Parser._parse_types


def _patched_parse_types(original_method):
    """Wrap _parse_types to handle BITMAP as a special case."""

    def wrapper(self, check_func=False, schema=False, allow_identifiers=True):
        # Check if current token is BITMAP (as an identifier)
        if self._match_text_seq("BITMAP"):
            # Create a DataType with BITMAP as a custom type
            # We'll map it to HLLSKETCH type internally since it's similar conceptually
            return exp.DataType(this=exp.DataType.Type.HLLSKETCH, nested=False)
        return original_method(self, check_func, schema, allow_identifiers)

    return wrapper


# Apply the wrapper to Doris and StarRocks parsers
Doris.Parser._parse_types = _patched_parse_types(original_doris_parse_types)
StarRocks.Parser._parse_types = _patched_parse_types(original_starrocks_parse_types)
