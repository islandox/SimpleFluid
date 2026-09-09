#!/usr/bin/env python3
"""Compare separate-process JSONL runs; bytes are decimal bytes, not GB/GiB."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = next(r for r in records if r['type'] == 'metadata')
    counts = defaultdict(int)
    stages = defaultdict(list)
    for record in records:
        if record['type'] != 'checkpoint':
            continue
        key = (record['rank'], record['stage'])
        index = counts[key]
        counts[key] += 1
        stages[(record['stage'], index)].append(record)
    if ('cleared', 0) not in stages:
        raise ValueError(f'{path}: incomplete or failed run (missing cleared checkpoint)')
    for stage, records in stages.items():
        if sorted(r['rank'] for r in records) != list(range(metadata['ranks'])):
            raise ValueError(f'{path}: missing/duplicate ranks at {stage}')
    return metadata, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    base_meta, baseline = load(args.baseline)
    new_meta, candidate = load(args.candidate)
    for key in ('commit', 'compiler', 'build', 'global_cells', 'ranks', 'scalar_bytes',
                'local_index_bytes', 'global_index_bytes', 'offset_bytes', 'kokkos',
                'tolerance', 'max_iterations', 'block_size', 'num_blocks', 'preconditioner'):
        if base_meta[key] != new_meta[key]:
            raise ValueError(f'incompatible {key}: {base_meta[key]} != {new_meta[key]}')
    comparisons = []
    for stage in baseline:
        old, new = baseline[stage], candidate[stage]
        metrics = {}
        for metric in ('graph_view_bytes', 'value_view_bytes', 'composite_scratch_payload_bytes',
                       'simultaneous_sum_rss_bytes', 'maximum_rank_peak_rss_bytes', 'seconds',
                       'iterations', 'true_relative_residual', 'continuity_l2'):
            # Payloads are disjoint rank-local allocations; RSS aggregates were
            # already reduced at the checkpoint. Wall time uses the slowest rank.
            reduce = sum if metric in ('graph_view_bytes', 'value_view_bytes',
                                       'composite_scratch_payload_bytes') else max
            a, b = reduce(r[metric] for r in old), reduce(r[metric] for r in new)
            metrics[metric] = {'baseline': a, 'candidate': b, 'difference': b - a,
                               'difference_percent': 100 * (b - a) / a if a else None}
        comparisons.append({'stage': stage[0], 'occurrence': stage[1], 'metrics': metrics})
    print(json.dumps({'baseline': str(args.baseline), 'candidate': str(args.candidate),
                      'baseline_metadata': base_meta, 'candidate_metadata': new_meta,
                      'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
