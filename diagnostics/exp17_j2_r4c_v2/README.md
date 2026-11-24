# R4-C v2 archived resource HOLD

This version preserves the v1 evidence and records a separately approved
increment of two A traversals, two N-prefix traversals, and six component
Autograd calls. The byte-identical bridge and fixtures reuse the original twelve
CPU Autograd validations; no new CPU Autograd is performed.

The nineteen pure-Python guard tests pass. Graph A now correctly distinguishes
one root module hook from two forward method dispatches, records its observation
before validation, and completes its A/N forward checks.

**Do not retry this frozen run.** Graph B's A traversal reached module19 when
the process RSS crossed the 20GiB ceiling. The ledger preserves the pending
`B_full` attempt. Graph B's N prefix and all model component Autograd calls were
never started. No model-gradient comparison is available.

The original checkpoint, production code and scientific results remain intact.
See the v2 handoff and the separately proposed storage/budget revision.
