# Transform this repo in place into the plugin

Rather than porting the fork onto upstream v30 first and extracting the
plugin afterwards, or starting a fresh plugin repo while keeping the fork
alive, this repository (already named `sqlglot-clickzetta`) transforms
directly: the vendored `sqlglot/` core is deleted, the ClickZetta dialect,
its tests, and the git history stay, and the package declares a narrow
dependency on released upstream sqlglot (`>=30.2,<31`).

## Consequences

- The v23→v30 dialect port happens once, inside the end-state repo — no
  monster merge whose obsoleted-patch conflict resolutions get thrown away,
  and no two-sources-of-truth drift with a live fork.
- The regression net is CI installing pinned upstream sqlglot and running
  upstream's own test suite alongside ours — with and without the compat
  module armed — plus our dialect suite. The "without compat" leg doubles
  as the graceful-degradation test required by ADR-0001.
- The fork branch (`clickzetta` as of the transformation commit) is frozen
  as a reference; fork wheels stop being built at cutover.
