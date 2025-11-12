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
# https://doris.apache.org/zh-CN/docs/4.x/table-design/data-model/unique
if not hasattr(exp, 'UniqueKeyProperty'):
    class UniqueKeyProperty(exp.Property):
        arg_types = {"expressions": True}


    # Register it in the exp module
    exp.UniqueKeyProperty = UniqueKeyProperty


# Add AggregateKeyProperty expression class for Doris AGGREGATE KEY
# https://doris.apache.org/zh-CN/docs/4.x/table-design/data-model/aggregate
if not hasattr(exp, 'AggregateKeyProperty'):
    class AggregateKeyProperty(exp.Property):
        arg_types = {"expressions": True}


    # Register it in the exp module
    exp.AggregateKeyProperty = AggregateKeyProperty

# Monkey patch Doris and StarRocks parsers to support UNIQUE KEY and BITMAP
original_doris_parse_create = Doris.Parser._parse_create
original_starrocks_parse_create = StarRocks.Parser._parse_create


def _parse_unique_key(self):
    """Parse UNIQUE KEY syntax."""
    self._match_text_seq("KEY")
    expressions = self._parse_wrapped_csv(self._parse_id_var, optional=False)
    return self.expression(exp.UniqueKeyProperty, expressions=expressions)


def _parse_aggregate_key(self):
    """Parse AGGREGATE KEY syntax.

    AGGREGATE KEY is a Doris-specific feature that defines which columns are key columns
    in an aggregate table model. Value columns can have aggregation types like:
    SUM, REPLACE, MAX, MIN, REPLACE_IF_NOT_NULL, HLL_UNION, BITMAP_UNION.

    Since ClickZetta doesn't support this, we parse it but will ignore it in the output.
    """
    self._match_text_seq("KEY")
    expressions = self._parse_wrapped_csv(self._parse_id_var, optional=False)
    return self.expression(exp.AggregateKeyProperty, expressions=expressions)


def _patched_doris_parse_create(self):
    """Patched Doris _parse_create to support UNIQUE KEY and AGGREGATE KEY.

    - UNIQUE KEY: Move from properties to schema (as constraint)
    - AGGREGATE KEY: Remove from properties (not supported in ClickZetta)
    """
    create = original_doris_parse_create(self)

    # Process properties
    if isinstance(create, exp.Create) and isinstance(create.this, exp.Schema):
        props = create.args.get("properties")
        if props:
            # Move UniqueKey from properties to schema (similar to PrimaryKey handling in StarRocks)
            unique_key = props.find(exp.UniqueKeyProperty)
            if unique_key:
                create.this.append("expressions", unique_key.pop())

            # Remove AggregateKey from properties (not supported in ClickZetta)
            aggregate_key = props.find(exp.AggregateKeyProperty)
            if aggregate_key:
                aggregate_key.pop()  # Just remove it, don't add to schema

    return create


def _patched_starrocks_parse_create(self):
    """Patched StarRocks _parse_create to support UNIQUE KEY and AGGREGATE KEY.

    - UNIQUE KEY: Move from properties to schema (as constraint)
    - AGGREGATE KEY: Remove from properties (not supported in ClickZetta)
    """
    create = original_starrocks_parse_create(self)

    # Process properties
    if isinstance(create, exp.Create) and isinstance(create.this, exp.Schema):
        props = create.args.get("properties")
        if props:
            # Move UniqueKey from properties to schema (similar to PrimaryKey handling)
            unique_key = props.find(exp.UniqueKeyProperty)
            if unique_key:
                create.this.append("expressions", unique_key.pop())

            # Remove AggregateKey from properties (not supported in ClickZetta)
            aggregate_key = props.find(exp.AggregateKeyProperty)
            if aggregate_key:
                aggregate_key.pop()  # Just remove it, don't add to schema

    return create


# Apply monkey patches
Doris.Parser._parse_create = _patched_doris_parse_create
Doris.Parser._parse_unique = _parse_unique_key
Doris.Parser._parse_aggregate = _parse_aggregate_key
StarRocks.Parser._parse_create = _patched_starrocks_parse_create
StarRocks.Parser._parse_unique = _parse_unique_key
StarRocks.Parser._parse_aggregate = _parse_aggregate_key

# Add UNIQUE and AGGREGATE to PROPERTY_PARSERS for both dialects
for dialect in [Doris, StarRocks]:
    if hasattr(dialect.Parser, 'PROPERTY_PARSERS'):
        dialect.Parser.PROPERTY_PARSERS = {
            **dialect.Parser.PROPERTY_PARSERS,
            "UNIQUE": lambda self: self._parse_unique(),
            "AGGREGATE": lambda self: self._parse_aggregate(),
        }

# Apply the wrapper to Doris and StarRocks parsers
# Doris.Parser._parse_types = _patched_parse_types(original_doris_parse_types)
# StarRocks.Parser._parse_types = _patched_parse_types(original_starrocks_parse_types)


# Add support for Doris aggregation type modifiers in column definitions
# In Doris AGGREGATE tables, value columns can have aggregation types like:
# SUM, REPLACE, MAX, MIN, REPLACE_IF_NOT_NULL, HLL_UNION, BITMAP_UNION
# These appear after the data type: cost BIGINT SUM DEFAULT '0'
# We need to skip these keywords by monkey patching _parse_column_def

# List of Doris aggregation type keywords
# These keywords appear after the data type in AGGREGATE KEY tables
DORIS_AGGREGATION_TYPES = {
    "SUM", "REPLACE", "MAX", "MIN",
    "REPLACE_IF_NOT_NULL", "HLL_UNION", "BITMAP", "BITMAP_UNION", "QUANTILE_UNION"
}

# Save original _parse_column_def methods
original_doris_parse_column_def = Doris.Parser._parse_column_def
original_starrocks_parse_column_def = StarRocks.Parser._parse_column_def


def _patched_parse_column_def_with_aggregation(original_method):
    """Wrap _parse_column_def to handle Doris aggregation type modifiers.

    After parsing the column type, we check if the next token is an aggregation
    type modifier (SUM, REPLACE, MAX, etc.) and skip it if present.
    """
    def wrapper(self, this):
        # If this is None, just use the original method
        # This happens for table-level constraints like PRIMARY KEY
        if this is None:
            return original_method(self, this)

        # First, parse the type using the standard method
        kind = self._parse_types(schema=True)

        # Check if next token is a Doris aggregation type modifier
        if self._curr and self._curr.text and self._curr.text.upper() in DORIS_AGGREGATION_TYPES:
            # Skip the aggregation type keyword
            self._advance()

        # Now parse the rest using the standard constraint parsing
        # This handles NOT NULL, DEFAULT, COMMENT, etc.
        constraints = []
        while True:
            constraint = self._parse_column_constraint()
            if not constraint:
                break
            constraints.append(constraint)

        # Return the ColumnDef expression
        return self.expression(
            exp.ColumnDef,
            this=this,
            kind=kind,
            constraints=constraints if constraints else None
        )

    return wrapper


# Apply the patched _parse_column_def
Doris.Parser._parse_column_def = _patched_parse_column_def_with_aggregation(original_doris_parse_column_def)
StarRocks.Parser._parse_column_def = _patched_parse_column_def_with_aggregation(original_starrocks_parse_column_def)



# ============================================================================
# Inverted Index Support for Doris and StarRocks
# ============================================================================
# Doris and StarRocks support inverted indexes with the syntax:
# INDEX idx_name(column) USING INVERTED PROPERTIES("key" = "value", ...)
#
# Since ClickZetta doesn't support inverted indexes, we:
# 1. Parse them in Doris/StarRocks (so no syntax errors)
# 2. Ignore them when generating ClickZetta SQL (via monkey patch)

# First, extend IndexConstraintOption to support inverted index properties
# We store all PROPERTIES as a dictionary in a new 'properties' arg
original_index_option_arg_types = exp.IndexConstraintOption.arg_types.copy()
exp.IndexConstraintOption.arg_types = {
    **original_index_option_arg_types,
    "properties": False,  # Dict of properties for INVERTED index
}

# Save original _parse_index_constraint methods
original_doris_parse_index_constraint = Doris.Parser._parse_index_constraint
original_starrocks_parse_index_constraint = StarRocks.Parser._parse_index_constraint


def _patched_parse_index_constraint_with_inverted(original_method):
    """Wrap _parse_index_constraint to handle USING INVERTED PROPERTIES(...).

    Doris inverted index syntax:
        INDEX idx_name(column) USING INVERTED PROPERTIES("key" = "value", ...) [COMMENT '...']

    We parse it as a regular IndexColumnConstraint with:
    - index_type: "INVERTED"
    - options: contains IndexConstraintOption with properties dict and optional comment
    """
    def wrapper(self, kind: t.Optional[str] = None) -> exp.IndexColumnConstraint:
        # Parse basic index structure: [kind] [INDEX] name (columns) [USING type]
        if kind:
            self._match_texts(("INDEX", "KEY"))

        # Parse index name
        this = self._parse_id_var(any_token=False)

        # Parse indexed columns first (before USING clause for Doris/StarRocks)
        # MySQL syntax: INDEX name USING type (columns)
        # Doris syntax: INDEX name (columns) USING type
        # We support both by checking for parenthesis first
        if self._match(TokenType.L_PAREN):
            # Doris-style: parse columns, then USING
            expressions = self._parse_csv(self._parse_ordered)
            self._match_r_paren()
            # Now check for USING clause
            index_type = self._match(TokenType.USING) and self._advance_any() and self._prev.text
        else:
            # MySQL-style: parse USING, then columns
            index_type = self._match(TokenType.USING) and self._advance_any() and self._prev.text
            expressions = self._parse_wrapped_csv(self._parse_ordered)

        # Parse options (COMMENT, KEY_BLOCK_SIZE, WITH PARSER, etc.)
        options = []

        # Check if this is an INVERTED index with PROPERTIES
        is_inverted = index_type and index_type.upper() == "INVERTED"

        if is_inverted:
            # Parse PROPERTIES if present
            if self._match_text_seq("PROPERTIES"):
                if self._match(TokenType.L_PAREN):
                    properties = {}
                    while True:
                        # Parse "key" = "value"
                        key_expr = self._parse_string()
                        if not key_expr:
                            break

                        # Extract string value from Literal expression
                        if isinstance(key_expr, exp.Literal):
                            key = key_expr.this
                        else:
                            key = str(key_expr)

                        self._match(TokenType.EQ)
                        value_expr = self._parse_string()
                        if value_expr:
                            # Extract string value from Literal expression
                            if isinstance(value_expr, exp.Literal):
                                value = value_expr.this
                            else:
                                value = str(value_expr)
                            properties[key] = value

                        # Check for comma separator
                        if not self._match(TokenType.COMMA):
                            break

                    self._match(TokenType.R_PAREN)

                    # Store properties in an IndexConstraintOption
                    if properties:
                        opt = exp.IndexConstraintOption(properties=properties)
                        options.append(opt)

            # Parse COMMENT if present (for inverted indexes)
            if self._match(TokenType.COMMENT):
                comment_expr = self._parse_string()
                if comment_expr:
                    opt = exp.IndexConstraintOption(comment=comment_expr)
                    options.append(opt)
        else:
            # Not an inverted index - return original parse
            return original_method(self, kind)

        return self.expression(
            exp.IndexColumnConstraint,
            this=this,
            expressions=expressions,
            kind=kind,
            index_type=index_type,
            options=options,
        )

    return wrapper


# Apply monkey patch to Doris and StarRocks parsers
Doris.Parser._parse_index_constraint = _patched_parse_index_constraint_with_inverted(
    original_doris_parse_index_constraint
)
StarRocks.Parser._parse_index_constraint = _patched_parse_index_constraint_with_inverted(
    original_starrocks_parse_index_constraint
)


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


# ============================================================================
# Doris-Specific Data Types
# ============================================================================

# Add custom data type classes for Doris-specific types that don't exist in SQLGlot

# LARGEINT: 128-bit signed integer (Doris-specific)
if not hasattr(exp.DataType.Type, 'LARGEINT'):
    exp.DataType.Type.LARGEINT = "LARGEINT"

# HLL: HyperLogLog type for approximate COUNT DISTINCT (Doris aggregate type)
if not hasattr(exp.DataType.Type, 'HLL'):
    exp.DataType.Type.HLL = "HLL"

# QUANTILE_STATE: Type for computing approximate quantiles (Doris aggregate type)
if not hasattr(exp.DataType.Type, 'QUANTILE_STATE'):
    exp.DataType.Type.QUANTILE_STATE = "QUANTILE_STATE"

# AGG_STATE: Generic aggregate function state type (Doris aggregate type)
if not hasattr(exp.DataType.Type, 'AGG_STATE'):
    exp.DataType.Type.AGG_STATE = "AGG_STATE"

# VARIANT: Dynamic data type for semi-structured data like JSON (Doris-specific)
if not hasattr(exp.DataType.Type, 'VARIANT'):
    exp.DataType.Type.VARIANT = "VARIANT"

# BITMAP
if not hasattr(exp.DataType.Type, 'BITMAP'):
    exp.DataType.Type.BITMAP = "BITMAP"


# Update the Doris Parser's _parse_types method to handle these custom types
original_doris_parse_types = Doris.Parser._parse_types

def _patched_doris_parse_types(self, check_func=False, schema=False, allow_identifiers=True):
    """Patched _parse_types to handle Doris-specific types."""
    # Check if current token is one of our custom Doris types
    if self._curr:
        type_text = self._curr.text.upper() if self._curr.text else ""

        # Create DataType with the enum Type value
        # IMPORTANT: Use exp.DataType.Type enum value, not string!
        # This ensures TYPE_MAPPING works correctly in generators.
        if type_text == "LARGEINT":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.LARGEINT)
        elif type_text == "HLL":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.HLL)
        elif type_text == "QUANTILE_STATE":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.QUANTILE_STATE)
        elif type_text == "AGG_STATE":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.AGG_STATE)
        elif type_text == "VARIANT":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.VARIANT)
        elif type_text == "BITMAP":
            self._advance()
            return self.expression(exp.DataType, this=exp.DataType.Type.BITMAP)

    # Fall back to original implementation
    return original_doris_parse_types(self, check_func=check_func, schema=schema, allow_identifiers=allow_identifiers)

Doris.Parser._parse_types = _patched_doris_parse_types
StarRocks.Parser._parse_types = _patched_doris_parse_types