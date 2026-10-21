"""Atomic, content-checked task artifacts; source photos are only read."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def fingerprint(path, check=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if check:
                check()
            digest.update(block)
    return digest.hexdigest()


def file_state(path):
    state = os.stat(path)
    return [state.st_size, state.st_mtime_ns, state.st_ctime_ns, state.st_dev, state.st_ino]


def gallery_hashes(root):
    return {str(path.relative_to(root)): fingerprint(path) for path in sorted(Path(root).rglob('*'))
            if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png')}


def atomic_json(path, value):
    path = Path(path)
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError('output_directory_symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def copy_verified(source, destination, digest):
    """Reuse an identical published file; refuse unrelated collisions."""
    source, destination = Path(source), Path(destination)
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError('output_directory_symlink')
    if fingerprint(source) != digest:
        raise ValueError('source_content_changed_before_copy')
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or fingerprint(destination) != digest:
            raise ValueError(f'output_content_conflict: {destination.name}')
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = stream.name
        shutil.copy2(source, temporary)
        if fingerprint(temporary) != digest or fingerprint(source) != digest:
            raise ValueError('source_content_changed_during_copy')
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        # Exclusive link publication prevents overwriting a concurrent writer.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or fingerprint(destination) != digest:
                raise ValueError(f'output_content_conflict: {destination.name}')
            return False
        return True
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
