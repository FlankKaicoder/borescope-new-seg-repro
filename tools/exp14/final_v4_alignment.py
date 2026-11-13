"""Process A8 repair returns and make one final pass over the authorized old Raw files."""
import argparse
from pathlib import Path

import cv2

from tools.exp14.final_v4 import A7_ROOT, frozen_plan
from tools.exp14.targeted_v3 import A6, PHASE
from tools.exp14.targeted_v3_alignment import process, rescue_old_raw

SUFFIX = "_A8ALIGN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    scope, _, _ = frozen_plan(root)
    cv2.setNumThreads(2)
    rescue_old_raw(root, scope, history_dirs=(PHASE, A6, A7_ROOT), suffix=SUFFIX,
                   fit_options={"require_target_support": True})
    process(root, args.follow, False, plan_loader=frozen_plan, rescue=False)


if __name__ == "__main__":
    main()
