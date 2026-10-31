#!/usr/bin/env python3
"""Engineering-only launcher for original scripts (zero changes to source code).

Optional --cpu-map-location maps serialized CUDA checkpoint tensors to CPU on
CPU-only hosts. Tensor values, model definitions, parameters and policy logic
remain unchanged. This is not an algorithm correction or a reproduction claim.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    'train2': 'train2.py',
    'train_diffusion': 'train_diffusion.py',
    'eval_td3': 'eval_td3.py',
}

def main():
    parser = argparse.ArgumentParser(description='Run ORIGINAL untouched scripts and pass through their CLI arguments.')
    parser.add_argument('--cpu-map-location', action='store_true', help='Opt-in: CPU-map CUDA-saved PyTorch tensors for CPU-only computers.')
    parser.add_argument('script', choices=tuple(SCRIPTS))
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ['--']:
        passthrough = passthrough[1:]
    script = ROOT/'data_collection'/SCRIPTS[args.script]
    # Original scripts assume top-level imports and local working directory.
    sys.path.insert(0, str(ROOT/'PVS'/'src'))
    sys.path.insert(0, str(ROOT/'data_collection'))
    os.chdir(ROOT/'data_collection')
    if args.cpu_map_location:
        import torch
        if torch.cuda.is_available():
            parser.error('--cpu-map-location is for CPU-only systems. Do not use it when CUDA is available.')
        original_load = torch.load
        def load_on_cpu(*pos, **kw):
            kw.setdefault('map_location', 'cpu')
            return original_load(*pos, **kw)
        torch.load = load_on_cpu
        print('[launch_original] CPU-only checkpoint device mapping enabled; algorithm source unchanged.',flush=True)
    sys.argv=[str(script), *passthrough]
    runpy.run_path(str(script),run_name='__main__')

if __name__ == '__main__':
    main()
