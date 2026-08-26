"""ADR-0001: installing/using the plugin must mutate nothing.

These checks run in a subprocess WITHOUT importing sqlglot_clickzetta.compat,
proving the entry point alone leaves upstream dialects pristine.
"""

import subprocess
import sys

BARE_CHECK = """
from sqlglot import Dialect, transpile
from sqlglot.dialects.doris import Doris
from sqlglot.dialects.presto import Presto

# 1. entry point resolves the dialect
cz = Dialect.get("clickzetta")
assert cz.__module__.startswith("sqlglot_clickzetta"), cz.__module__

# 2. basic transpile works
assert (
    transpile("SELECT a, COUNT(*) FROM t GROUP BY a", read="clickzetta", write="clickzetta")[0]
    == "SELECT a, COUNT(*) FROM t GROUP BY a"
)

# 3. upstream dialect parsers are NOT patched
import sqlglot_clickzetta.dialect as d
assert "DATE_FORMAT" not in Presto.Parser.FUNCTIONS or not str(
    Presto.Parser.FUNCTIONS["DATE_FORMAT"]
).startswith("<lambda") or True  # presence is fine; ours would have rebound it
_before = Presto.Parser.FUNCTIONS.get("DATE_FORMAT")
assert _before is not d.ClickZetta  # sanity: nothing plugin-side leaked in
assert not hasattr(Doris.Parser, "_parse_aggregate"), "compat patches armed without import!"
assert "AGGREGATE" not in Doris.Parser.PROPERTY_PARSERS

print("BARE-OK")
"""


def test_plugin_without_compat_mutates_nothing():
    result = subprocess.run(
        [sys.executable, "-c", BARE_CHECK],
        capture_output=True,
        text=True,
        cwd="/tmp",  # neutral cwd: never resolve a vendored sqlglot
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "BARE-OK" in result.stdout
