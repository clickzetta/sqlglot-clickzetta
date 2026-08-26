# Cross-dialect patches live in an explicit opt-in compat module

Converting the sqlglot fork to a plugin means we can no longer modify core
sqlglot, but ClickZetta transpilation requires `read="doris"` /
`read="presto"` (and other source dialects) to parse ClickZetta-specific
extensions. Surviving patches live in a separate `compat` module that
consumers import explicitly — installing the plugin or using the
`clickzetta` dialect alone mutates nothing. (Rejected: applying patches at
entry-point load, which would silently alter foreign dialect parses for
public-PyPI strangers.)

Function *routing* to compatibility builtins (AES_*_MYSQL,
TRUNCATE_PRESTO, DATE_FORMAT_MYSQL for the MySQL family and ClickHouse,
the ClickHouse function mappings) does NOT patch readers: it lives in the
dialect generator, keyed on the source-dialect stamp that compat sets on
every parsed SELECT. Formats recover exactly by re-inverting through the
*source* dialect's inverse time mapping; ClickHouse formats pass through
verbatim. (Initially rejected wholesale as "a larger port for little gain";
probe evidence during the port showed renames and format round-trips are
tractable, and this shrank compat's cross-dialect footprint accordingly.)

Two remaps provably cannot move generator-side and stay read-side:
Postgres/Redshift `TO_CHAR` → `DATE_FORMAT_PG` (PG's parse normalizes
dd/DD and yyyy/YYYY case variants to one python token — the user's
original casing, which the engine builtin needs, is unrecoverable from the
AST), and Presto-DLC 2-arg `date_add` (parse tolerance — the reader must
accept the syntax at all).

## Consequences

- Without `import sqlglot_clickzetta.compat`, there is no source-dialect
  stamp, so the generator's routing branches take their default path —
  un-compat'd transpilation produces valid, less-tuned SQL, never wrong
  SQL. Tested explicitly (tests/test_bare.py).
- Once compat is imported, all dialects in the process are patched —
  coexisting code sees it. Named, documented, never accidental.
- The patches depend on upstream Parser internals (`_parse_create`,
  `_parse_column_def`, `PROPERTY_PARSERS`, `_parse_types` signatures), so
  upstream refactors can break compat between versions. This is the main
  reason the plugin pins a narrow sqlglot version range.
- Patches already obsoleted by upstream (UNIQUE/AGGREGATE KEY, MySQL
  DATE_ADD intervals, `JSONEXTRACTSTRING`, `HLL`/`VARIANT` types,
  `DECIMAL32/64/128` tokens) are deleted rather than ported; upstreamable
  feature gaps get PR'd to tobymao/sqlglot and deleted once merged.

