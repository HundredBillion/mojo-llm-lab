"""Fetch checksum-pinned public GGML inputs; never execute downloaded files."""
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def fetch(url, path, expected):
    if path.is_file() and digest(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        with urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if digest(temporary) != expected:
            raise ValueError(f'Checksum mismatch: {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    versions = json.loads((ROOT / 'versions.json').read_text())
    target = ROOT / 'build/upstream'
    archive = target / versions['archive']
    release = versions['engine_release']
    fetch(f'https://github.com/ggml-org/llama.cpp/releases/download/{release}/{archive.name}',
          archive, versions['archive_sha256'])
    if archive.stat().st_size != versions['archive_bytes']:
        raise ValueError('Incorrect archive length')
    # Always extract from the verified archive, so stale edited binaries are not reused.
    with tarfile.open(archive) as source:
        source.extractall(target, filter='data')
    revision = versions['engine_revision']
    for name, checksum in versions['header_sha256'].items():
        fetch(f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{revision}/ggml/include/{name}',
              target / 'headers' / name, checksum)
    print(f'Verified {release} binaries and headers in build/upstream/')


if __name__ == '__main__':
    main()
