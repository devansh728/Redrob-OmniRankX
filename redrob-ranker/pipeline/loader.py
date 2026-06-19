import gzip
import json
import os

def stream_candidates(path):
    if not os.path.exists(path):
        return

    is_gz = path.endswith(".gz")
    open_fn = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"
    encoding = "utf-8"

    with open_fn(path, mode, encoding=encoding) as f:
        first_char = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break

    with open_fn(path, mode, encoding=encoding) as f:
        if first_char == "[":
            data = json.load(f)
            for item in data:
                yield item
        else:
            for line in f:
                line_str = line.strip()
                if line_str:
                    yield json.loads(line_str)

def load_all(path):
    return list(stream_candidates(path))
