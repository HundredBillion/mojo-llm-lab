"""Paired ABBA diagnostics; these do not replace the existing control gate."""
import math
import statistics


def paired_abba(samples, candidate):
    if len(samples) != 32:
        raise ValueError('Require eight complete ABBA rounds')
    rounds = {}
    for sample in samples:
        value = sample['gpu_us_per_operation']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('GPU times must be finite and positive')
        rounds.setdefault(sample['round'], []).append(sample)
    if set(rounds) != set(range(8)):
        raise ValueError('Missing or unexpected round identifiers')
    ratios = []
    for index in range(8):
        rows = rounds[index]
        if [row['mode'] for row in rows] != ['ggml', candidate, candidate, 'ggml']:
            raise ValueError('Require recorded ABBA order within every round')
        a = (rows[0]['gpu_us_per_operation'] + rows[3]['gpu_us_per_operation']) / 2
        b = (rows[1]['gpu_us_per_operation'] + rows[2]['gpu_us_per_operation']) / 2
        ratios.append(a/b)
    return {'baseline_over_candidate_round_ratios': ratios,
            'geometric_mean_ratio': math.exp(statistics.mean(math.log(r) for r in ratios)),
            'scope': 'Paired diagnostic only; does not override control rejection or establish stationarity.'}
