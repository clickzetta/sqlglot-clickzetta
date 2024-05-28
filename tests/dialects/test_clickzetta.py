
from tests.dialects.test_dialect import Validator


class TestClickZetta(Validator):
    dialect = "spark"

    def test_reserved_keyword(self):
        self.validate_all(
            "SELECT user.id from t as user",
            write={
                "clickzetta": "SELECT `user`.id FROM t AS `user`",
            },
        )

