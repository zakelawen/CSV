#!/usr/bin/env python
"""
Wrapper for Step9 CSV training on Gemma-2-9B only.

This file reuses scripts/step9_train_csv.py, but replaces MODEL_REGISTRY
with a single model key: gemma9b.

Usage:
  python scripts/step9_train_csv_gemma9b.py --model gemma9b --str_layers 0,1,2 --cls_layers 2,4,6,-1
"""

import os
import step9_train_csv as base


base.MODEL_REGISTRY = {
    "gemma9b": {
        "hf_name": os.environ.get(
            "GEMMA9B_MODEL_PATH",
            "google/gemma-2-9b",
        ),
    },
}


if __name__ == "__main__":
    base.main()