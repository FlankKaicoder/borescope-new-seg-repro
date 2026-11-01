# Exp13 cloud source archive and publication tools

Start at [the unified index](../../docs/exp13_cloud/EXP13_CLOUD_INDEX.md).

Stage subdirectories contain exact existing code snapshots. They are not
automatically portable runtime installations. Consult
[reproduction notes](../../docs/exp13_cloud/REPRODUCIBILITY.md) before any future
separately authorized private-workspace use. No runtime command is authorized
by this Git sync, and copying code must not reset historical POST budgets.

`sync_public_records.py` packages only explicit code/prompt/report paths and
private-artifact hashes. It never imports experiment code, decrypts a key,
decodes an image or accesses a network. It refuses overwriting prior outputs.

`audit_publication.py` inspects actual staged blobs against the exact public
allowlist, validates code provenance, compiles source without execution,
checks frozen-document/registry prefixes, and checks new sensitive paths and
credential patterns. It does not inspect historical TEST images or read keys.

The publishing path is separate from G1B/G1E live code and G1F DPAPI loaders.
Their presence is for reproducibility and independent review, not execution.
