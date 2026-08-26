# Cross-dialect patches live in an explicit opt-in compat module

Converting the sqlglot fork to a plugin means we can no longer modify core
sqlglot, but ClickZetta transpilation requires `read="doris"` /
`read="presto"` (and other source dialects) to parse ClickZetta-specific
extensions (Presto-DLC syntax, Doris table features, routing to
compatibility builtins like `DATE_FORMAT_MYSQL`, source-dialect stamping for
dialect-aware generation). We keep these as monkey-patches of upstream
dialect classes, but they live in a separate `compat` module that consumers
import explicitly — installing the plugin or using the `clickzetta` dialect
alone mutates nothing. (Considered and rejected: applying patches at
entry-point load, which would silently alter foreign dialect parses for
public-PyPI strangers; and absorbing everything into the ClickZetta
parser/generator, which is a larger port for little gain.)

## Consequences

- Without `import sqlglot_clickzetta.compat`, the ClickZetta generator's
  source-dialect-aware branches (keyed on the Select `dialect` stamp) take
  their default path. This degradation must stay graceful — un-compat'd
  transpilation should produce valid, less-tuned SQL, never wrong SQL.
  Tested explicitly.
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

