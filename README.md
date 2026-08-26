# sqlglot-clickzetta

A [ClickZetta](https://www.clickzetta.com) SQL dialect plugin for
[sqlglot](https://github.com/tobymao/sqlglot) — parse, transpile, and generate
ClickZetta SQL, including ClickZetta's acceptance of Presto-DLC, Doris,
StarRocks, ClickHouse and MySQL-flavored input.

This package is a plugin (requires `sqlglot>=30.2,<31`), not a fork: upstream
sqlglot stays a normal dependency.

## Install

```bash
pip install sqlglot-clickzetta
```

The `clickzetta` dialect registers itself automatically through sqlglot's
plugin entry points — no extra imports needed:

```python
import sqlglot

sqlglot.parse_one("SELECT UNIQEXACT(id) FROM t", read="clickzetta")
sqlglot.transpile("SELECT DATE_ADD(ts, INTERVAL 1 DAY)", read="starrocks", write="clickzetta")[0]
# SELECT TIMESTAMP_OR_DATE_ADD('DAY', 1, ts)
```

## The compat module (opt-in)

ClickZetta's engine accepts SQL written in other engines' dialects, extended
with ClickZetta-specific syntax. Supporting that requires patching how
*source* dialects parse. Those patches are **not** armed by installation —
installing or using this plugin never changes how `doris`, `presto`, or any
other dialect parses. To transpile ClickZetta-extended input, import compat
explicitly:

```python
import sqlglot_clickzetta.compat  # arms cross-dialect patches

sqlglot.transpile(
    "CREATE TABLE t (k BIGINT, v BIGINT SUM) AGGREGATE KEY(k) DISTRIBUTED BY HASH(k) BUCKETS 10",
    read="doris",
    write="clickzetta",
)[0]
```

What compat arms:

- **Source-dialect stamping** — every parsed `SELECT` records its read
  dialect, so the ClickZetta generator can vary output by input origin
  (e.g. `UNIX_TIMESTAMP` format semantics differ between MySQL-family and
  Presto sources).
- **Presto-DLC** — 2-arg `date_add(date, days)` alongside standard 3-arg Presto.
- **Doris / StarRocks table features** upstream sqlglot lacks: `AGGREGATE KEY`,
  column aggregation modifiers, inverted indexes, `AUTO PARTITION BY
  RANGE/LIST`, `LARGEINT`/`HLL`/`BITMAP`/`QUANTILE_STATE`/`AGG_STATE` types.
- **`TO_CHAR` verbatim remap** (Postgres/Redshift) — the engine's
  `DATE_FORMAT_PG` builtin needs the user's format string untouched, and
  PG's parse destroys the original letter casing.

Compatibility-builtin *routing* for the rest — `AES_DECRYPT` →
`AES_DECRYPT_MYSQL`, `TRUNCATE` → `TRUNCATE_PRESTO`, the MySQL-family
`DATE_FORMAT` → `DATE_FORMAT_MYSQL` (with format round-trip through the
source dialect's mapping), and the ClickHouse function mappings — lives in
the ClickZetta generator itself, keyed on the stamp; it needs no reader
patches.

Once compat is armed, all dialects in the process are patched — import it
once, early, in processes that transpile into ClickZetta. See
[ADR-0001](docs/adr/0001-opt-in-compat-module-for-cross-dialect-patches.md)
for the reasoning.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

CI runs the plugin suite across Python 3.10–3.13 against both ends of the
supported sqlglot range, plus upstream sqlglot's own dialect suite with the
plugin installed — proving the plugin never contaminates other dialects.

Architecture decisions are recorded in [docs/adr/](docs/adr/); project
terminology in [CONTEXT.md](CONTEXT.md).

## License

MIT — inherited from sqlglot.
