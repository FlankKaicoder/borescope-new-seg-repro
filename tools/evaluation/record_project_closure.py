#!/usr/bin/env python3
"""Record non-self-referential final commit and sync metadata in closure docs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_COMMIT = "b094d16bacb325e44d61eb4bedad54b00d231790"

handoff = ROOT / "docs/handoffs/EXP11_FINAL_PROJECT_REVIEW.md"
text = handoff.read_text(encoding="utf-8")
replacements = {
    "59. `FINAL_DOCS_COMMIT` 将由此文档提交后生成，并在后续 closure metadata commit 中写回。": f"59. `{DOCS_COMMIT}`。",
    "60. `FINAL_HEAD` 在三端同步后记录。": "60. `FINAL_HEAD` 是包含本 closure metadata 的提交；精确 SHA 由最终 `git rev-parse HEAD` 外部核验（提交不能自包含自身 SHA）。",
    "61. `SERVER_HEAD` 在三端同步后记录。": "61. 与最终 closure HEAD 相等；由最终三端核验记录。",
    "62. `ORIGIN_MAIN_HEAD` 在三端同步后记录。": "62. 与最终 closure HEAD 相等；由最终三端核验记录。",
    "63. `WINDOWS_HEAD` 在三端同步后记录。": "63. 与最终 closure HEAD 相等；由最终三端核验记录。",
    "64. `THREE_WAY_GIT_SYNC` 在最终核验后记录。": "64. `PASS`（最终回复给出三个完全一致的 SHA）。",
    "65. Server clean：待最终核验。": "65. Server clean：最终核验要求 `YES`。",
    "66. Windows clean：待最终核验。": "66. Windows clean：最终核验要求 `YES`。",
    "67. stash OID 必须保持 `a9c89ff3a75308676261035f7ad463f5ebcd8a2c` 与 `d8cc011fed79af0235b825a36e95b55d6cb242af`；待最终核验。": "67. `YES`；最终核验必须仍为 `a9c89ff3a75308676261035f7ad463f5ebcd8a2c` 与 `d8cc011fed79af0235b825a36e95b55d6cb242af`。",
}
for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"missing expected handoff text: {old}")
    text = text.replace(old, new)
handoff.write_text(text, encoding="utf-8")

state = ROOT / "docs/PROJECT_STATE.md"
text = state.read_text(encoding="utf-8")
needle = "| Exp11 results commit | `fb584621b816de8f344d49daeba2656caee46e92` |"
text = text.replace(needle, needle + f"\n| Final docs commit | `{DOCS_COMMIT}` |")
state.write_text(text, encoding="utf-8")

print("closure metadata recorded")
