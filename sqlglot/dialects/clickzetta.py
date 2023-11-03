from __future__ import annotations
import typing as t

from sqlglot import exp, transforms
from sqlglot.dialects.spark import Spark
from sqlglot.tokens import Tokenizer, TokenType
from sqlglot.dialects.dialect import (
    rename_func,
)

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
    # in MaxCompute, datetime(col) is a alias of cast(col as datetime)
    if expression.this.upper() == 'DATETIME':
        return f"{self.sql(expression.expressions[0])}::TIMESTAMP"
    elif expression.this.upper() == 'GETDATE':
        return f"CURRENT_TIMESTAMP()"

    # return as it is
    args = ", ".join(self.sql(e) for e in expression.expressions)
    return f"{expression.this}({args})"

class ClickZetta(Spark):

    NULL_ORDERING = "nulls_are_small"

    class Tokenizer(Spark.Tokenizer):
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "CREATE USER": TokenType.COMMAND,
            "DROP USER": TokenType.COMMAND,
            "SHOW USER": TokenType.COMMAND,
            "REVOKE": TokenType.COMMAND,
        }

    class Parser(Spark.Parser):
        pass

    class Generator(Spark.Generator):

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

        }

        TRANSFORMS = {
            **Spark.Generator.TRANSFORMS,
            exp.DefaultColumnConstraint: lambda self, e: '',
            exp.OnUpdateColumnConstraint: lambda self, e: '',
            exp.AutoIncrementColumnConstraint: lambda self, e: '',
            exp.CollateColumnConstraint: lambda self, e: '',
            exp.CharacterSetColumnConstraint: lambda self, e: '',
            exp.Create: transforms.preprocess([_transform_create]),
            exp.GroupConcat: _groupconcat_to_wmconcat,
            exp.AesDecrypt: rename_func("AES_DECRYPT_MYSQL"),
            exp.CurrentTime: lambda self, e: "DATE_FORMAT(NOW(),'HH:mm:ss')",
            exp.Anonymous: _anonymous_func,
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
               }:
                return type_sql
            return super().datatype_sql(expression)

        def tochar_sql(self, expression: exp.ToChar) -> str:
            this = expression.args.get('this')
            format = expression.args.get('format')
            if format:
                format_str = str(format).replace('mm', 'MM').replace('mi', 'mm')
                return f"DATE_FORMAT({self.sql(this)}, {self.sql(format_str)})"

            return super().tochar_sql(expression)

        def maybe_comment(self, sql: str, expression: exp.Expression | None = None, comments: List[str] | None = None) -> str:
            comments = (
                ((expression and expression.comments) if comments is None else comments)  # type: ignore
                if self.comments
                else None
            )

            if not comments or isinstance(expression, self.EXCLUDE_COMMENTS):
                return sql

            comments_sql = "\n".join(
                f"-- {self.pad_comment(comment)}" for comment in comments if comment
            )

            if not comments_sql:
                return sql

            if isinstance(expression, self.WITH_SEPARATED_COMMENTS):
                return (
                    f"{self.sep()}{comments_sql}{sql}"
                    if sql[0].isspace()
                    else f"{comments_sql}{self.sep()}{sql}"
                )

            return f"{sql} {comments_sql}"
