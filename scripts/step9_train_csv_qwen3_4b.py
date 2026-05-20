#!/usr/bin/env python
"""
Wrapper for Step9 CSV training on Qwen3-4B-Base only.

This file reuses scripts/step9_train_csv.py, but replaces MODEL_REGISTRY
with a single model key: qwen3_4b.

Usage:
  python scripts/step9_train_csv_qwen3_4b.py --model qwen3_4b --str_layer 0 --cls_layer 2
  python scripts/step9_train_csv_qwen3_4b.py --model qwen3_4b --str_layers 0,1,2 --cls_layers 2,4,6,-1

Override the model snapshot path via the QWEN3_4B_MODEL_PATH environment variable.
"""

import os
import step9_train_csv as base


base.MODEL_REGISTRY = {
    "qwen3_4b": {
        "hf_name": os.environ.get(
            "QWEN3_4B_MODEL_PATH",
            "Qwen/Qwen3-4B-Base",
        ),
    },
}


if __name__ == "__main__":
    base.main()