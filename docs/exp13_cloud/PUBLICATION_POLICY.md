# Public and private artifact policy

The repository remains public. Existing tracked images and commits are retained
as explicitly accepted by the user for this task; this policy is prospective
and does not claim to purge or make old history private.

| Publish after explicit audit | Keep in the controlled local asset store |
|---|---|
| Credential-free experiment source and DPAPI-loader source | API keys, DPAPI ciphertext, decrypted values, environment dumps |
| Fixed Prompt text and protocol summaries | Industrial originals, annotation JSON, ROI, masks |
| Sanitized stage/Gate records and aggregate metrics | Raw API outputs, review sheets, thumbnails, contact sheets |
| SHA256 and opaque asset IDs, non-secret relative paths | Full request/response logs, signed image URLs |
| Reproduction instructions and generic safety mechanisms | Paid-call approval receipts, production STARTED markers, M1 private A/B maps |

All historical private artifacts remain under `${PRIVATE_ASSET_ROOT}` (the
existing Windows controlled directory). No whole-directory copy is allowed.
Only the explicit source-file map in `sync_public_records.py` may publish
code, prompts, and selected report text. Image bytes and original annotation
content are never copied by that tool; hash reads do not regenerate assets.

The exact `public_allowlist.txt` is the only set permitted in this sync's index.
The staged audit inspects the actual Git index and diff, not merely filesystem
ignore behavior. New publication permits UTF-8 text only, blocks unexpected
paths, binary data, credential patterns, annotation structures, signed URLs,
private map/approval/STARTED files, and validates original-vs-copy provenance.

`.gitignore` provides additional scoped runtime/media exclusions. It cannot
remove tracked files or protect old commits. Never stage with broad add commands;
use only explicitly reviewed whitelist paths. No LFS or archive may bypass the
ban on new industrial media.

Runtime source contains environment-variable names and confirmation literals;
these are mechanisms, not credentials or an actual paid-request authorization.
No approved Gate receipt is published. Copying source must not reset a private
request budget or create permission for another historical request.

Inherited historical code may write raw request/response metadata or signed
output URLs into its private runtime directories. Such logs must stay private
even if the generating source is public. DPAPI code decryption is never invoked
by this synchronization or its publication safety audit.
