# Reports

Dated, one-off audits of the repository. Each describes the commit it audited,
and its findings are history: some were fixed afterwards and some were decided
against, as the notes at the top of each report say. None of them is a list of
current defects.

| Report | What it audited |
| --- | --- |
| [RELEASE_AUDIT.md](RELEASE_AUDIT.md) | What to fix before the first PyPI release (2026-09-02) |
| [API_AUDIT_core_building_blocks.md](API_AUDIT_core_building_blocks.md) | `Animatable.__init__`, `Mob.__init__`, `Surface` and `BezierCircuitCubic` against `DOCSTRINGS.md` |
| [UX_AUDIT_2026-09-10.md](UX_AUDIT_2026-09-10.md) | The public authoring surface, and the fixes made alongside it |
| [UX_AUDIT_VALIDATION_2026-09-10.md](UX_AUDIT_VALIDATION_2026-09-10.md) | The test runs that validated the UX audit's fixes |

For work that is still open, read [TODO.md](../../TODO.md). For how to cut a
release, read [RELEASE_RUNBOOK.md](../../RELEASE_RUNBOOK.md).
