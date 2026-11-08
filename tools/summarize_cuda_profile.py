"""Aggregate Nsight kernel events without interpreting them as application throughput."""
from contextlib import closing
import argparse
import gzip
import json
from pathlib import Path
import sqlite3


def summarize(events):
    groups = {}
    total = 0
    count = 0
    for event in events:
        name = event['kernel']
        start, end = event['start_ns'], event['end_ns']
        grid, block = event['grid'], event['block']
        if not isinstance(name, str) or not name:
            raise ValueError('Missing kernel name')
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError('Invalid kernel interval')
        if any(len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape) for shape in (grid, block)):
            raise ValueError('Invalid launch dimensions')
        key = (name, tuple(grid), tuple(block))
        row = groups.setdefault(key, {'kernel': name, 'grid': list(grid), 'block': list(block), 'calls': 0, 'gpu_ns': 0})
        row['calls'] += 1
        row['gpu_ns'] += end-start
        total += end-start
        count += 1
    if not count:
        raise ValueError('No kernel events')
    rows = sorted(groups.values(), key=lambda r: (-r['gpu_ns'], r['kernel'], r['grid'], r['block']))
    for row in rows:
        row['share_percent'] = 100*row['gpu_ns']/total
    return {'scope': 'Summed recorded GPU kernel durations; includes warmup unless separately filtered. Not wall-clock time or production throughput.',
            'kernel_calls': count, 'total_gpu_ns': total, 'groups': rows}


def sqlite_events(path):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
        query = '''SELECT s.value,k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
                   FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id
                   ORDER BY k.start,k.end'''
        for name, start, end, *shape in connection.execute(query):
            yield {'kernel': name, 'start_ns': start, 'end_ns': end, 'grid': shape[:3], 'block': shape[3:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Nsight SQLite or exported .jsonl.gz kernel events')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events-output', type=Path, help='Optional gzip JSON-lines export containing only kernel names, timestamps and launch dimensions')
    args = parser.parse_args()
    if args.input.name.endswith('.jsonl.gz'):
        with gzip.open(args.input, 'rt') as source:
            events = [json.loads(line) for line in source]
    else:
        events = list(sqlite_events(args.input))
    result = summarize(events)
    if args.events_output:
        with gzip.open(args.events_output, 'xt') as output:
            for event in events:
                output.write(json.dumps(event)+'\n')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'kernel_calls': result['kernel_calls'], 'total_gpu_ns': result['total_gpu_ns']}))


if __name__ == '__main__':
    main()
