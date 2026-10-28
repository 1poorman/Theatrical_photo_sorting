"""Verify that the Git index excludes local data and generated runtime artifacts."""
import argparse
from collections import Counter
import json
from pathlib import PurePosixPath
import subprocess


LOCAL_ROOTS = {'data', 'weights', 'outputs', 'logs', '.agents', '.aws', '.codex', '.vscode'}
GENERATED_SUFFIXES = {'.pyc', '.pyo', '.sqlite', '.sqlite3', '.db', '.npz', '.npy', '.faiss',
                      '.onnx', '.safetensors', '.engine'}


def excluded(path):
    parts = PurePosixPath(path).parts
    return (parts[0] in LOCAL_ROOTS or '__pycache__' in parts or
            path.startswith(('tests/local/', 'core_modules/face_recognition/trt_cache/')) or
            PurePosixPath(path).suffix in GENERATED_SUFFIXES or
            PurePosixPath(path).name == '.env' or
            (PurePosixPath(path).name.startswith('.env.') and
             PurePosixPath(path).name != '.env.example'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    files = subprocess.check_output(['git', 'ls-files', '-z']).decode().rstrip('\0').split('\0')
    files = [path for path in files if path]
    violations = [path for path in files if excluded(path)]
    result = {'passed': not violations, 'tracked_files': len(files),
              'tracked_roots': dict(sorted(Counter(path.split('/')[0] for path in files).items())),
              'excluded_files_still_tracked': len(violations), 'violations': violations}
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else
          f"Git scope: {'PASS' if result['passed'] else 'FAIL'}; "
          f"{len(files)} tracked files, {len(violations)} excluded files still tracked")
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
