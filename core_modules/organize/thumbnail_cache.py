"""Bounded content-addressed previews; original images remain read-only."""
from contextlib import contextmanager
import fcntl
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile

from PIL import Image, ImageOps, __version__ as pillow_version

from .feature_cache import ContentHasher, signature


MAGIC = b'THEATRE-THUMB-1\n'
ENTRY_NAME = re.compile(r'[0-9a-f]{64}\.thumb\Z')


class ThumbnailCache:
    def __init__(self, root, max_bytes=256 * 1024 * 1024, max_entries=4096):
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self.hasher = ContentHasher()

    def _read(self, path):
        try:
            # Never follow links when reading or pruning the cache directory.
            if path.is_symlink() or path.stat().st_size > self.max_bytes:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        header = len(MAGIC) + 32
        if not data.startswith(MAGIC) or len(data) <= header:
            return None
        jpeg = data[header:]
        return jpeg if hashlib.sha256(jpeg).digest() == data[len(MAGIC):header] else None

    @contextmanager
    def _publication_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.publication.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _publish(self, target, jpeg):
        payload = MAGIC + hashlib.sha256(jpeg).digest() + jpeg
        if len(payload) > self.max_bytes or self.max_entries == 0:
            return
        with self._publication_lock():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as stream:
                    temporary = stream.name
                    stream.write(payload)
                os.replace(temporary, target)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            entries = []
            for path in self.root.iterdir():
                if ENTRY_NAME.fullmatch(path.name) and not path.is_symlink() and path.is_file():
                    stat = path.stat()
                    entries.append((stat.st_mtime_ns, path, stat.st_size))
            count, total = len(entries), sum(row[2] for row in entries)
            for _, path, length in sorted(entries):
                if count <= self.max_entries and total <= self.max_bytes:
                    break
                path.unlink()
                count -= 1
                total -= length

    @staticmethod
    def _render(path, size):
        with Image.open(path) as source:
            source.draft('RGB', (size, size))
            ImageOps.exif_transpose(source, in_place=True)
            source.thumbnail((size, size))
            picture = source if source.mode == 'RGB' else source.convert('RGB')
            try:
                buffer = BytesIO()
                picture.save(buffer, format='JPEG', quality=85)
                return buffer.getvalue()
            finally:
                if picture is not source:
                    picture.close()

    def preview(self, path, size):
        content, version = self.hasher.fingerprint(path)
        key = signature({'content': content, 'size': size, 'recipe': 'draft-exif-thumbnail-rgb-v1',
                         'format': 'JPEG', 'quality': 85, 'pillow': pillow_version})
        target = self.root / (key + '.thumb')
        jpeg = self._read(target)
        hit = jpeg is not None
        if not hit:
            jpeg = self._render(path, size)
        # Metadata is a write guard; complete source bytes are checked on hits
        # as well, including equal-size rewrites with a restored mtime.
        if self.hasher.fingerprint(path) != (content, version):
            raise ValueError('source_changed_during_thumbnail')
        if not hit:
            self._publish(target, jpeg)
        return jpeg, key
