# Exp13.0 VLM Semantic Audit

## Purpose

Establish a TRAIN-only, human-reviewable input package for assessing whether a vision-language model can describe real seven-class borescope defects in a fixed structured format suitable for future prompt construction.

## Scope

This stage prepares 20 Crack and 20 corrosion images, polygon context, mask overlays, contact sheets, a fixed JSON schema, and a prompt template. It does not call a VLM and does not produce semantic model outputs.

## Data boundary

The source is the frozen dataset v1 TRAIN split only. No VAL or TEST images, labels, metrics, or selection decisions were accessed. The source dataset and labels remain read-only; generated material is stored under this experiment's results directory.

## Explicit exclusions

No diffusion, Stable Diffusion, ControlNet, synthetic image generation, SSL training, YOLO training, model download, or GPU training is part of Exp13.0.

## Gate boundary

Exp13.0 is preparation/audit infrastructure only. Exp13.1 Diffusion remains blocked until this package is reviewed and the semantic-audit Gate is explicitly authorized.

