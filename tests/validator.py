import unittest

from sqlglot import ErrorLevel, UnsupportedError, parse_one
from sqlglot.parser import logger as parser_logger


class Validator(unittest.TestCase):
    """Minimal parity oracle ported from the fork's tests/dialects/test_dialect.py."""

    dialect = None

    def parse_one(self, sql, **kwargs):
        return parse_one(sql, read=self.dialect, **kwargs)

    def validate_identity(
        self, sql, write_sql=None, pretty=False, check_command_warning=False, identify=False
    ):
        if check_command_warning:
            with self.assertLogs(parser_logger) as cm:
                expression = self.parse_one(sql)
                assert f"'{sql[:100]}' contains unsupported syntax" in cm.output[0]
        else:
            expression = self.parse_one(sql)

        self.assertEqual(
            write_sql or sql, expression.sql(dialect=self.dialect, pretty=pretty, identify=identify)
        )
        return expression

    def validate_all(self, sql, read=None, write=None, pretty=False, identify=False):
        """
        Validate that:
        1. Everything in `read` transpiles to `sql`
        2. `sql` transpiles to everything in `write`

        Args:
            sql (str): Main SQL expression
            read (dict): Mapping of dialect -> SQL
            write (dict): Mapping of dialect -> SQL
            pretty (bool): prettify both read and write
            identify (bool): quote identifiers in both read and write
        """
        expression = self.parse_one(sql)

        for read_dialect, read_sql in (read or {}).items():
            with self.subTest(f"{read_dialect} -> {sql}"):
                self.assertEqual(
                    parse_one(read_sql, read_dialect).sql(
                        self.dialect,
                        unsupported_level=ErrorLevel.IGNORE,
                        pretty=pretty,
                        identify=identify,
                    ),
                    sql,
                )

        for write_dialect, write_sql in (write or {}).items():
            with self.subTest(f"{sql} -> {write_dialect}"):
                if write_sql is UnsupportedError:
                    with self.assertRaises(UnsupportedError):
                        expression.sql(write_dialect, unsupported_level=ErrorLevel.RAISE)
                else:
                    self.assertEqual(
                        expression.sql(
                            write_dialect,
                            unsupported_level=ErrorLevel.IGNORE,
                            pretty=pretty,
                            identify=identify,
                        ),
                        write_sql,
                    )
