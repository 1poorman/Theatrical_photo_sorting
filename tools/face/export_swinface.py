"""Export compact SwinFace recognition + age weights, preserving the original."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core_modules.face_recognition.swinface.compact import PRECISIONS, export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'weights/swinface/checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--output', type=Path, required=True, help='New .pt file; existing files are never replaced')
    parser.add_argument('--precision', choices=PRECISIONS, default='fp16')
    parser.add_argument('--torch-threads', type=int, default=4)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error('torch-threads must be positive')
    import torch
    torch.set_num_threads(args.torch_threads)
    report = export_checkpoint(args.source, args.output, args.precision)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
