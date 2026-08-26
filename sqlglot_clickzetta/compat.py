"""ClickZetta cross-dialect compatibility patches.

Importing this module arms monkey-patches that make upstream sqlglot dialects
accept ClickZetta-extended input: Presto-DLC syntax, Doris/StarRocks table
features upstream lacks, function routing to ClickZetta compatibility
builtins (DATE_FORMAT_MYSQL, ...), and source-dialect stamping that the
ClickZetta generator keys on. Installing the plugin or using the
``clickzetta`` dialect alone mutates nothing — only an explicit

    import sqlglot_clickzetta.compat

arms these (ADR-0001, docs/adr/0001-opt-in-compat-module-for-cross-dialect-patches.md).

Once armed, all dialects in this process are patched. Patches obsoleted by
upstream sqlglot (UNIQUE/AGGREGATE-key property parsing covered natively,
MySQL DATE_ADD intervals, JSONEXTRACTSTRING, HLL/VARIANT/DECIMAL32 types)
are intentionally not carried here; upstreamable gaps live here only until
their upstream PRs land.
"""

from __future__ import annotations

import typing as t

from sqlglot import exp
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

from sqlglot_clickzetta.dialect import AggregateKeyProperty


# ---------------------------------------------------------------------------
# Source-dialect stamping
# ---------------------------------------------------------------------------

_parse_select = getattr(Parser, "_parse_select")


def preprocess_parse_select(self, *args, **kwargs):
    expression = _parse_select(self, *args, **kwargs)
    if not expression:
        return expression
    # Stamp the read dialect onto the Select so the ClickZetta generator can
    # vary output by input origin (e.g. UNIX_TIMESTAMP format semantics).
    # v23 fork used the dialect module's last segment; v30 parsers hold a
    # dialect *instance*, and this package's own module name would stamp as
    # "DIALECT" — the class name is stable for upstream and plugin alike.
    read_dialect = self.dialect.__class__.__name__.upper()
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


# ---------------------------------------------------------------------------
# Compatibility-builtin routing and Presto-DLC function parsing
# ---------------------------------------------------------------------------

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

# Clickhouse's toDateTime(expr[, timezone]) parameter expr supports String, Int, Date or DateTime.
# To adapt to multiple types, we use the cast function for conversion.
# Notice: This will cause TODATETIME -> CAST related cases to fail.
ClickHouse.Parser.FUNCTIONS["TODATETIME"] = lambda args: exp.cast(
    seq_get(args, 0), exp.DataType.Type.DATETIME
)
ClickHouse.Parser.FUNCTIONS["TODATE"] = lambda args: exp.cast(
    seq_get(args, 0), exp.DataType.Type.DATE
)


# ---------------------------------------------------------------------------
# Doris / StarRocks AGGREGATE KEY
# ---------------------------------------------------------------------------

original_doris_parse_create = Doris.Parser._parse_create
original_starrocks_parse_create = StarRocks.Parser._parse_create


def _parse_aggregate_key(self):
    """Parse AGGREGATE KEY syntax.

    AGGREGATE KEY is a Doris-specific feature that defines which columns are key columns
    in an aggregate table model. Value columns can have aggregation types like:
    SUM, REPLACE, MAX, MIN, REPLACE_IF_NOT_NULL, HLL_UNION, BITMAP_UNION.

    Since ClickZetta doesn't support this, we parse it but will ignore it in the output.
    """
    self._match_text_seq("KEY")
    expressions = self._parse_wrapped_csv(self._parse_id_var, optional=False)
    return self.expression(AggregateKeyProperty(expressions=expressions))


def _patched_doris_parse_create(self):
    """Patched Doris _parse_create to support AGGREGATE KEY (UNIQUE KEY is native upstream).

    - UNIQUE KEY: Move from properties to schema (as constraint) if upstream
      parsing left it in properties
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
            aggregate_key = props.find(AggregateKeyProperty)
            if aggregate_key:
                aggregate_key.pop()  # Just remove it, don't add to schema

    return create


def _patched_starrocks_parse_create(self):
    """Patched StarRocks _parse_create, same treatment as Doris."""
    create = original_starrocks_parse_create(self)

    if isinstance(create, exp.Create) and isinstance(create.this, exp.Schema):
        props = create.args.get("properties")
        if props:
            unique_key = props.find(exp.UniqueKeyProperty)
            if unique_key:
                create.this.append("expressions", unique_key.pop())

            aggregate_key = props.find(AggregateKeyProperty)
            if aggregate_key:
                aggregate_key.pop()

    return create


# Apply monkey patches. UNIQUE KEY property parsing is native in upstream
# sqlglot >= 28; only AGGREGATE KEY routing is ours.
Doris.Parser._parse_create = _patched_doris_parse_create
Doris.Parser._parse_aggregate = _parse_aggregate_key
StarRocks.Parser._parse_create = _patched_starrocks_parse_create
StarRocks.Parser._parse_aggregate = _parse_aggregate_key

# Add AGGREGATE to PROPERTY_PARSERS for both dialects (UNIQUE is native)
for dialect in [Doris, StarRocks]:
    if hasattr(dialect.Parser, 'PROPERTY_PARSERS'):
        dialect.Parser.PROPERTY_PARSERS = {
            **dialect.Parser.PROPERTY_PARSERS,
            "AGGREGATE": lambda self: self._parse_aggregate(),
        }


# ---------------------------------------------------------------------------
# Doris aggregation type modifiers in column definitions
# ---------------------------------------------------------------------------

# In Doris AGGREGATE tables, value columns can have aggregation types after
# the data type: cost BIGINT SUM DEFAULT '0'. We skip these keywords by
# monkey patching _parse_column_def.

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
    def wrapper(self, this, **kwargs):
        # If this is None, just use the original method
        # This happens for table-level constraints like PRIMARY KEY
        if this is None:
            return original_method(self, this, **kwargs)

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
            exp.ColumnDef(this=this, kind=kind, constraints=constraints if constraints else None)
        )

    return wrapper


# Apply the patched _parse_column_def
Doris.Parser._parse_column_def = _patched_parse_column_def_with_aggregation(original_doris_parse_column_def)
StarRocks.Parser._parse_column_def = _patched_parse_column_def_with_aggregation(original_starrocks_parse_column_def)


# ---------------------------------------------------------------------------
# Inverted Index Support for Doris and StarRocks
# ---------------------------------------------------------------------------
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
            exp.IndexColumnConstraint(
                this=this,
                expressions=expressions,
                kind=kind,
                index_type=index_type,
                options=options,
            )
        )

    return wrapper


# Apply monkey patch to Doris and StarRocks parsers
Doris.Parser._parse_index_constraint = _patched_parse_index_constraint_with_inverted(
    original_doris_parse_index_constraint
)
StarRocks.Parser._parse_index_constraint = _patched_parse_index_constraint_with_inverted(
    original_starrocks_parse_index_constraint
)


# ---------------------------------------------------------------------------
# AUTO PARTITION BY RANGE/LIST and PARTITION BY RANGE/LIST in Doris/StarRocks
# ---------------------------------------------------------------------------

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
                schema = self.expression(exp.Schema(expressions=partition_expr))
            else:
                # Single expression - wrap in Schema with single expression
                schema = self.expression(exp.Schema(expressions=[partition_expr]))
            return self.expression(exp.PartitionedByProperty(this=schema))
        else:
            # Not RANGE or LIST - restore position and use original parser
            self._retreat(start_index)
            return original_method(self)

    return wrapper


# Save original methods
original_doris_parse_auto_property = Doris.Parser._parse_auto_property
original_starrocks_parse_auto_property = StarRocks.Parser._parse_auto_property

# Apply the wrapper to _parse_auto_property (with PARTITION BY matching)
Doris.Parser._parse_auto_property = _create_partition_parser_wrapper(
    original_doris_parse_auto_property, match_partition_by=True
)
StarRocks.Parser._parse_auto_property = _create_partition_parser_wrapper(
    original_starrocks_parse_auto_property, match_partition_by=True
)

# v30 parses `PARTITION BY RANGE(...)` natively into PartitionByRangeProperty
# (which the ClickZetta generator does not emit) and crashes on empty
# definition lists `()`. The property-parser keys ("PARTITION BY",
# "PARTITIONED BY", ...) no longer route through _parse_partitioned_by, so we
# intercept at PROPERTY_PARSERS dispatch: RANGE/LIST partitions collapse to a
# plain PartitionedByProperty and definitions are skipped, matching the fork.
_PARTITION_PROPERTY_KEYS = ("PARTITION", "PARTITION BY", "PARTITIONED BY", "PARTITIONED_BY")


def _parse_partition_property_compat(original_parser):
    def handler(self):
        if self._match(TokenType.RANGE) or self._match(TokenType.LIST):
            partition_expr = self._parse_wrapped_csv(self._parse_field)

            # Skip the partition definition list — all shapes (FIXED RANGE
            # `VALUES [(..), (..))`, LESS THAN, FROM..TO, VALUES IN, empty).
            if self._match(TokenType.L_PAREN):
                outer_depth = 1
                inner_depth = -1
                while outer_depth > 0 and self._curr is not None:
                    if self._match(TokenType.L_BRACKET):
                        inner_depth = 0
                    elif self._match(TokenType.R_BRACKET):
                        pass
                    elif self._match(TokenType.L_PAREN):
                        if inner_depth >= 0:
                            inner_depth += 1
                        else:
                            outer_depth += 1
                    elif self._match(TokenType.R_PAREN):
                        if inner_depth > 0:
                            inner_depth -= 1
                        elif inner_depth == 0:
                            inner_depth = -1
                        else:
                            outer_depth -= 1
                    else:
                        self._advance()

            expressions = (
                partition_expr if isinstance(partition_expr, list) else [partition_expr]
            )
            return self.expression(
                exp.PartitionedByProperty(this=exp.Schema(expressions=expressions))
            )
        return original_parser(self)

    return handler


for _dialect in (Doris, StarRocks):
    _originals = {k: _dialect.Parser.PROPERTY_PARSERS[k] for k in _PARTITION_PROPERTY_KEYS}
    _dialect.Parser.PROPERTY_PARSERS = {
        **_dialect.Parser.PROPERTY_PARSERS,
        **{k: _parse_partition_property_compat(orig) for k, orig in _originals.items()},
    }


# ---------------------------------------------------------------------------
# Doris-Specific Data Types
# ---------------------------------------------------------------------------

# Add custom data type enum members for Doris-specific types upstream lacks.
# Upstream already ships HLL and VARIANT; these guards keep compat forward-
# compatible as the remaining types land upstream.

# LARGEINT: 128-bit signed integer (Doris-specific)
if not hasattr(exp.DataType.Type, 'LARGEINT'):
    exp.DataType.Type.LARGEINT = "LARGEINT"

# QUANTILE_STATE: Type for computing approximate quantiles (Doris aggregate type)
if not hasattr(exp.DataType.Type, 'QUANTILE_STATE'):
    exp.DataType.Type.QUANTILE_STATE = "QUANTILE_STATE"

# AGG_STATE: Generic aggregate function state type (Doris aggregate type)
if not hasattr(exp.DataType.Type, 'AGG_STATE'):
    exp.DataType.Type.AGG_STATE = "AGG_STATE"

# BITMAP
if not hasattr(exp.DataType.Type, 'BITMAP'):
    exp.DataType.Type.BITMAP = "BITMAP"

# HLL: HyperLogLog type for approximate COUNT DISTINCT (Doris/StarRocks
# aggregate type) — not in upstream v30 either.
if not hasattr(exp.DataType.Type, 'HLL'):
    exp.DataType.Type.HLL = "HLL"

# The dialect's TYPE_MAPPING carries string keys for these types (matching the
# v23 fork, where runtime-added members were plain strings on older Pythons).
# On v30 they are real enum members, so register member-keyed mappings too —
# string keys never match member lookups in the generator.
from sqlglot_clickzetta.dialect import ClickZetta

ClickZetta.Generator.TYPE_MAPPING.update({
    exp.DataType.Type.BITMAP: "BITMAP",
    exp.DataType.Type.LARGEINT: "BIGINT",
    exp.DataType.Type.QUANTILE_STATE: "BINARY",
    exp.DataType.Type.AGG_STATE: "BINARY",
    exp.DataType.Type.HLL: "BINARY",
})


# Update the Doris Parser's _parse_types method to handle these custom types
original_doris_parse_types = Doris.Parser._parse_types

def _patched_doris_parse_types(self, check_func=False, schema=False, allow_identifiers=True, with_collation=False):
    """Patched _parse_types to handle Doris-specific types."""
    # Check if current token is one of our custom Doris types
    if self._curr:
        type_text = self._curr.text.upper() if self._curr.text else ""

        # Create DataType with the enum Type value
        # IMPORTANT: Use exp.DataType.Type enum value, not string!
        # This ensures TYPE_MAPPING works correctly in generators.
        if type_text == "LARGEINT":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.LARGEINT))
        elif type_text == "HLL":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.HLL))
        elif type_text == "QUANTILE_STATE":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.QUANTILE_STATE))
        elif type_text == "AGG_STATE":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.AGG_STATE))
        elif type_text == "VARIANT":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.VARIANT))
        elif type_text == "BITMAP":
            self._advance()
            return self.expression(exp.DataType(this=exp.DataType.Type.BITMAP))

    # Fall back to original implementation
    return original_doris_parse_types(self, check_func=check_func, schema=schema, allow_identifiers=allow_identifiers, with_collation=with_collation)

Doris.Parser._parse_types = _patched_doris_parse_types
StarRocks.Parser._parse_types = _patched_doris_parse_types
