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
from sqlglot.tokens import TokenType


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
def _presto_date_add_parser(args: t.List) -> exp.DateAdd:
    """Support both 2-arg and 3-arg versions of date_add for presto-dlc.

    2-arg version (like MySQL): date_add(date, days)
    3-arg version (standard Presto): date_add(unit, amount, date)
    """
    if len(args) == 2:
        # 2-arg version: date_add(date, days) - assume unit is DAY
        return exp.DateAdd(
            this=args[0],
            expression=args[1],
            unit=exp.Var(this="DAY")
        )
    elif len(args) == 3:
        # 3-arg version: date_add(unit, amount, date)
        return exp.DateAdd(
            this=args[2],
            expression=args[1],
            unit=args[0]
        )
    else:
        # Fallback for unexpected arg counts
        return None

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

# Override date_add for Presto-based dialects to support both 2 and 3 parameters
for dialect in [Presto, Trino, Athena]:
    dialect.Parser.FUNCTIONS["DATE_ADD"] = _presto_date_add_parser
    # Convert Presto's TRUNCATE to TRUNCATE_PRESTO to distinguish from ClickZetta's TRUNCATE
    dialect.Parser.FUNCTIONS["TRUNCATE"] = lambda args: exp.Anonymous(
        this="TRUNCATE_PRESTO", expressions=args
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


# Add support for AUTO PARTITION BY RANGE/LIST and PARTITION BY RANGE/LIST in Doris/StarRocks
# Common function to handle partition parsing logic
def _create_partition_parser_wrapper(original_method, match_partition_by=False):
    """Create a wrapper for partition parsing methods.

    This is a generic function used by both _parse_auto_property and _parse_partitioned_by.
    doris manual-partitioning: https://doris.apache.org/zh-CN/docs/3.x/table-design/data-partitioning/manual-partitioning
    doris dynamic-partitioning: https://doris.apache.org/zh-CN/docs/3.x/table-design/data-partitioning/dynamic-partitioning
    doris auto-partitioning: https://doris.apache.org/zh-CN/docs/3.x/table-design/data-partitioning/auto-partitioning

    Args:
        original_method: The original parser method to wrap
        match_partition_by: If True, match PARTITION BY token first (for _parse_auto_property)
                          If False, PARTITION BY has already been consumed (for _parse_partitioned_by)

    Handles:
    - AUTO PARTITION BY RANGE(expr)() - extract expr only
    - AUTO PARTITION BY LIST(col1, col2, ...)() - extract column list only
    - PARTITION BY RANGE(expr)(...) - extract expr, skip all partition definitions
    - PARTITION BY LIST(col1, col2)(...) - extract columns, skip all partition definitions

    Supports all Doris partition definition types:
    - FIXED RANGE: VALUES [("start"), ("end"))
    - LESS THAN: VALUES LESS THAN ("value")
    - BATCH RANGE: FROM (start) TO (end) INTERVAL ...
    - MULTI RANGE: Multiple FROM...TO clauses
    - LIST: VALUES IN (...)
    - NULL partitions
    """
    def wrapper(self):
        # Save current position for potential fallback
        start_index = self._index

        # Check if this is PARTITION BY (for AUTO PARTITION case)
        if match_partition_by:
            if not self._match(TokenType.PARTITION_BY):
                # Not PARTITION BY - use original method
                return original_method(self)

        # Look ahead to see if this is RANGE or LIST
        partition_type = None
        if self._match(TokenType.RANGE):
            partition_type = "RANGE"
        elif self._match(TokenType.LIST):
            partition_type = "LIST"

        if partition_type:
            # Parse the partition expression/columns in parentheses
            # For RANGE: PARTITION BY RANGE(expr) - extract expr
            # For LIST: PARTITION BY LIST(col1, col2, ...) - extract columns
            partition_expr = self._parse_wrapped_csv(self._parse_field)

            # Skip any partition definitions that follow
            # They can be:
            # 1. (PARTITION p1 VALUES [...), (...)) - FIXED RANGE (left-closed, right-open)
            # 2. (PARTITION p1 VALUES LESS THAN (...)) - LESS THAN
            # 3. (FROM (...) TO (...) INTERVAL ...) - BATCH RANGE
            # 4. (FROM...TO..., FROM...TO..., PARTITION p VALUES [...)) - MULTI RANGE
            # 5. (PARTITION p1 VALUES IN (...)) - LIST
            # 6. Empty () for dynamic partitioning
            #
            # Special handling for FIXED RANGE notation: VALUES [(...), (...))
            # The bracket [ with paren ) means left-closed, right-open interval
            # Example: VALUES [('2017-01-01'), ('2017-02-01'))
            #   - [              # starts the interval, inner_depth = 0
            #   - (              # is the start value left, inner_depth += 1
            #   - '2017-01-01')  # is the first value, inner_depth -= 1
            #   - ,('2017-02-01') # is the end value
            #   - )              # closes the interval (matches with [), inner_depth = -1
            if self._match(TokenType.L_PAREN):
                outer_depth = 1  # Track outer partition definitions depth
                inner_depth = -1  # Track depth inside [...) interval

                while outer_depth > 0 and not self._curr is None:
                    if self._match(TokenType.L_BRACKET):
                        # Start of left-closed, right-open interval: [
                        inner_depth = 0  # Now we're in [...) interval, reset to 0
                    elif self._match(TokenType.R_BRACKET):
                        # Right bracket in FIXED RANGE - shouldn't happen
                        pass
                    elif self._match(TokenType.L_PAREN):
                        # Left paren - could be:
                        # 1. Start of value tuple inside [...) like ('2017-01-01')
                        # 2. Regular nested paren in other partition types
                        if inner_depth >= 0:
                            # We're inside a [...) interval, track inner depth
                            inner_depth += 1
                        else:
                            # Regular paren
                            outer_depth += 1
                    elif self._match(TokenType.R_PAREN):
                        # Right paren - need to determine what it closes
                        if inner_depth > 0:
                            # This closes a paren inside [...) like the ) in ('2017-01-01')
                            inner_depth -= 1
                        elif inner_depth == 0:
                            # This is the ) that closes [...) interval
                            # Reset inner_depth to -1 to indicate we're not in interval anymore
                            inner_depth = -1
                        else:
                            # Regular paren, part of outer structure
                            outer_depth -= 1
                    else:
                        self._advance()

            # Return PartitionedByProperty with just the columns/expression
            # Always wrap in a Schema for consistency with ClickZetta expectations
            if isinstance(partition_expr, list):
                # Multiple columns or single column - wrap in a Schema
                schema = self.expression(exp.Schema, expressions=partition_expr)
            else:
                # Single expression - wrap in Schema with single expression
                schema = self.expression(exp.Schema, expressions=[partition_expr])
            return self.expression(exp.PartitionedByProperty, this=schema)
        else:
            # Not RANGE or LIST - restore position and use original parser
            self._retreat(start_index)
            return original_method(self)

    return wrapper


# Save original methods
original_doris_parse_auto_property = Doris.Parser._parse_auto_property
original_starrocks_parse_auto_property = StarRocks.Parser._parse_auto_property
original_doris_parse_partitioned_by = Doris.Parser._parse_partitioned_by
original_starrocks_parse_partitioned_by = StarRocks.Parser._parse_partitioned_by

# Apply the wrapper to _parse_auto_property (with PARTITION BY matching)
Doris.Parser._parse_auto_property = _create_partition_parser_wrapper(
    original_doris_parse_auto_property, match_partition_by=True
)
StarRocks.Parser._parse_auto_property = _create_partition_parser_wrapper(
    original_starrocks_parse_auto_property, match_partition_by=True
)

# Apply the wrapper to _parse_partitioned_by (PARTITION BY already consumed)
Doris.Parser._parse_partitioned_by = _create_partition_parser_wrapper(
    original_doris_parse_partitioned_by, match_partition_by=False
)
StarRocks.Parser._parse_partitioned_by = _create_partition_parser_wrapper(
    original_starrocks_parse_partitioned_by, match_partition_by=False
)
