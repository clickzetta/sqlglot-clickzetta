from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.dialect import (
    approx_count_distinct_sql,
    arrow_json_extract_sql,
    parse_timestamp_trunc,
    rename_func,
)
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import seq_get
from sqlglot.tokens import Token, TokenType

class StarRocks(MySQL):
    class Tokenizer(MySQL.Tokenizer):
        KEYWORDS = {
            **MySQL.Tokenizer.KEYWORDS,
            "DECIMAL64": TokenType.DECIMAL,
            "DECIMAL128": TokenType.BIGDECIMAL,
        }

    class Parser(MySQL.Parser):
        FUNCTIONS = {
            **MySQL.Parser.FUNCTIONS,
            "DATE_TRUNC": parse_timestamp_trunc,
            "DATEDIFF": lambda args: exp.DateDiff(
                this=seq_get(args, 0), expression=seq_get(args, 1), unit=exp.Literal.string("DAY")
            ),
            "DATE_DIFF": lambda args: exp.DateDiff(
                this=seq_get(args, 1), expression=seq_get(args, 2), unit=seq_get(args, 0)
            ),
            "REGEXP": exp.RegexpLike.from_arg_list,
        }

        PROPERTY_PARSERS = {
            **MySQL.Parser.PROPERTY_PARSERS,
            "DISTRIBUTED": lambda self: self._parse_distributed_by(),
            "PROPERTIES": lambda self: self._parse_wrapped_csv(self._parse_property),
            "DUPLICATE": lambda self: self._parse_duplicate(),
            "PARTITION BY": lambda self: self._parse_partitioned_by(),
        }


        def _parse_distributed_by(self) -> exp.DistributedByProperty:
            self._match_text_seq("BY")
            self._match_text_seq("HASH")

            self._match_l_paren()
            expressions = self._parse_csv(self._parse_column)
            self._match_r_paren()

            if self._match_text_seq("SORTED", "BY"):
                self._match_l_paren()
                sorted_by = self._parse_csv(self._parse_ordered)
                self._match_r_paren()
            else:
                sorted_by = None

            self._match_text_seq("BUCKETS")
            buckets = self._parse_number()

            return self.expression(
                exp.DistributedByProperty,
                expressions=expressions,
                sorted_by=sorted_by,
                buckets=buckets,
            )

        def _parse_duplicate(self):
            self._match_text_seq("DUPLICATE")
            self._match_text_seq("KEY")
            self._match_l_paren()
            expressions = self._parse_csv(self._parse_column)
            self._match_r_paren()

        def _parse_partitioned_by(self) -> exp.PartitionedByProperty:
            self._match(TokenType.EQ)
            expp = self.expression(
                exp.PartitionedByProperty,
                this=self._parse_schema() or self._parse_bracket(self._parse_field()),
            )
            if self._match(TokenType.L_PAREN):
                self._parse_partition_value()

            return expp

        def _parse_partition_value(self):
            if self._match_text_seq("PARTITION"):
                self._parse_var()
                self._match_text_seq("VALUES")
                self._parse_partition_value_range()
                if self._match(TokenType.R_PAREN):
                    return
                else:
                    self._match(TokenType.COMMA)
                    self._parse_partition_value()

        def _parse_partition_value_range(self):
            self._match_set([TokenType.L_PAREN, TokenType.L_BRACKET])
            self._match_l_paren()
            self._parse_string()
            self._match_r_paren()
            self._match(TokenType.COMMA)
            self._parse_var()
            self._match_l_paren()
            self._parse_string()
            self._match_r_paren()
            self._match_set([TokenType.R_PAREN, TokenType.R_BRACKET])

    class Generator(MySQL.Generator):
        CAST_MAPPING = {}

        TYPE_MAPPING = {
            **MySQL.Generator.TYPE_MAPPING,
            exp.DataType.Type.TEXT: "STRING",
            exp.DataType.Type.TIMESTAMP: "DATETIME",
            exp.DataType.Type.TIMESTAMPTZ: "DATETIME",
        }

        TRANSFORMS = {
            **MySQL.Generator.TRANSFORMS,
            exp.ApproxDistinct: approx_count_distinct_sql,
            exp.DateDiff: lambda self, e: self.func(
                "DATE_DIFF", exp.Literal.string(e.text("unit") or "DAY"), e.this, e.expression
            ),
            exp.JSONExtractScalar: arrow_json_extract_sql,
            exp.JSONExtract: arrow_json_extract_sql,
            exp.RegexpLike: rename_func("REGEXP"),
            exp.StrToUnix: lambda self, e: f"UNIX_TIMESTAMP({self.sql(e, 'this')}, {self.format_time(e)})",
            exp.TimestampTrunc: lambda self, e: self.func(
                "DATE_TRUNC", exp.Literal.string(e.text("unit")), e.this
            ),
            exp.TimeStrToDate: rename_func("TO_DATE"),
            exp.UnixToStr: lambda self, e: f"FROM_UNIXTIME({self.sql(e, 'this')}, {self.format_time(e)})",
            exp.UnixToTime: rename_func("FROM_UNIXTIME"),
        }

        TRANSFORMS.pop(exp.DateTrunc)
