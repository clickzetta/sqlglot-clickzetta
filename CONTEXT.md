# sqlglot-clickzetta

A ClickZetta SQL dialect for sqlglot, being converted from a fork of upstream
sqlglot into a plugin package that depends on released upstream sqlglot.

## Language

**ClickZetta dialect**:
The sqlglot dialect that reads and writes ClickZetta SQL.
_Avoid_: clickzetta parser (ambiguous with the engine's own parser)

**Source dialect**:
The dialect a SQL string is written in (`transpile(read=...)`), as opposed to
the ClickZetta dialect it is being generated into. ClickZetta output can vary
depending on the source dialect.
_Avoid_: input dialect, foreign dialect

**Presto-DLC**:
ClickZetta's Presto-compatible input mode: standard Presto syntax plus
MySQL-style extensions (e.g. 2-arg `date_add(date, days)`). Not real Presto;
upstream sqlglot will never support it.
_Avoid_: presto (when DLC is meant), DLC

**Compatibility builtin**:
A ClickZetta engine function that emulates another engine's function under a
suffixed name (e.g. `DATE_FORMAT_MYSQL`, `DATE_FORMAT_PG`, `TRUNCATE_PRESTO`).
Real engine functions — routing source-dialect calls to them is a faithful
rename, not a semantic transform.

**Compat module**:
The plugin's `compat` — an explicit opt-in module arming the surviving
cross-dialect patches: source-dialect stamping, Presto-DLC `date_add`
tolerance, Doris/StarRocks table-feature gaps, and the `TO_CHAR` verbatim
remap. Compatibility-builtin *routing* for everything else lives in the
dialect generator, keyed on the stamp. Installing the plugin or using the
`clickzetta` dialect alone mutates nothing.
_Avoid_: settings (the old filename `local_clickzetta_settings.py` is
historical and misleading), lazy patching (rejected: surprises public-PyPI
strangers)

**Cross-dialect patch**:
A mutation of another dialect's Parser class. Obsoleted ones (~40% of
`local_clickzetta_settings.py`) get deleted against current upstream;
upstreamable ones (Doris/ClickHouse feature gaps) become upstream PRs;
ClickZetta-policy ones live in the compat module.
