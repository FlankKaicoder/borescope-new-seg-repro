# R4-C v1 archived diagnostic attempt — HOLD

These are the exact three Python files used by the first R4-C attempt. They are
outside `tools/` so the frozen production source catalog is unchanged.

**Do not launch or retry this version.** CPU derivative fixtures passed (12
Autograd calls). One A traversal and one N-prefix traversal returned, then the
diagnostic's root hook counter assertion failed. Graph B and all model component
Autograd calls remain unexecuted. The existing single-use directory and ledger
must not be reset or reused.

`BaseModel.loss` calls `self.forward(batch["img"])` directly. That method call
bypasses `Module.__call__`; the root module pre-hook does not observe a second
`tensor` entry. The v1 assertion incorrectly conflates method dispatches and
module hook events. The actual counter values were not saved at failure; this
explanation is a post-hoc static call-chain finding, not a backfilled record.

An additional, separate shutdown audit bug rejected `dir_fd`-relative temporary
cleanup inside the settings directory. The original evidence remains intact.

See the root R4-C report and the result directory's repair/reexecution proposal.
The failure is in this diagnostic implementation and provides no new evidence
about the magnitude of J2's actual detach gradient effect.
