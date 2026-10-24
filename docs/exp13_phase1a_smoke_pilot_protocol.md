# Exp13 Phase 1-A smoke and pilot protocol

This document defines a future, separately gated runtime sequence. Phase 1-A itself performs zero model calls.

Smoke is capped at four local calls: one representative ROI from a single-ROI source and one from a multi-ROI source, each at two strengths. The pilot proposal contains 21 TRAIN sources and 27 ROIs, hence 54 ROI-level calls at two strengths. The revised reserve is 16 calls. The empirical mean 100-source extension is 284 total local calls; the empirical p90 scenario is 400 and exceeds the 300 all-local cap, so production size is conditional rather than guaranteed.

Before any future call, the runner must verify the frozen candidate configuration, provenance, geometry invariants, and budget counters. It must stop on model loading failure, non-finite output, geometry mismatch, composition hold, unresolved FOV risk, or cap exceedance. No VAL/TEST image may enter smoke, pilot, retry, threshold selection, or model selection.
