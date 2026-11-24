# Frozen R4-C v3 diagnostic

This one-shot run completed 2 A traversals, 2 N-prefix traversals and 6 component
Autograd calls. No optimizer, VAL, TEST or new CPU Autograd was used. Twelve
previous CPU derivative validations were reused with identical bridge/fixtures.

Graph A saves tensors losslessly to an independent SHA-checked disk store;
Graph B saves on CPU. Both forward graphs are bitwise equal before backward.
YOLO and SupCon8 gradients are identical; Pair4 differs substantially.
The 19 guard and 23 storage tests passed before execution.

`run.py` is an archived single-use entrypoint, not a retry or training interface.
`analyze.py` is a subsequent offline NumPy-only audit and was not deployed as part
of the numerical process. See the v3 handoff for identities, budgets and limits.
The endpoint contrast does not establish the cause of historical BN mean drift.
