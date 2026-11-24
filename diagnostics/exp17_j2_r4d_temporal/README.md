# R4-D offline temporal investigation

`directions.py ROOT` reads22 identity-bound full checkpoint dictionaries and six
previously saved R4-C gradient arrays. It performs CPU tensor-to-NumPy algebra
under explicit model/Autograd/optimizer/CUDA guards. No model is instantiated.

`temporal.py` joins existing loss/probe/BN/Conv/optimizer/formal VAL tables.
Run its `--early-boundaries` supplement only after the first pass. Original
inputs are read-only; outputs are exclusive new files.

`summarize.py` independently matches direction magnitudes to existing R3 tables,
then creates public summaries and scientific figures. It does not import Torch.

All scripts are archived reproduction records, not training interfaces. The
remote first-reader failure and the temporal source revision are retained in
private evidence and source receipts. Historical parameter displacement is not
a reconstruction of historical component gradients. See the R4-D handoff.
