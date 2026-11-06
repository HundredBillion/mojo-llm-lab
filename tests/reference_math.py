# This mathematical float64 oracle does not replace the actual CUDA fast-math baseline.
import math

HEAD_DIM = 128
HEADS = 32
SECTIONS = (24, 20, 20, 0)
BASE = 5_000_000.0
EPSILON = 1e-6


def position_axis(pair):
    sector = pair % sum(SECTIONS)
    if sector % 3 == 1 and sector < 3 * SECTIONS[1]:
        return 1
    if sector % 3 == 2 and sector < 3 * SECTIONS[2]:
        return 2
    if sector % 3 == 0 and sector < 3 * SECTIONS[0]:
        return 0
    return 3


def reference_head(q, gamma, positions):
    if len(q) != HEAD_DIM or len(gamma) != HEAD_DIM or len(positions) != 4:
        raise ValueError("wrong reference shape")
    reciprocal_norm = 1.0 / math.sqrt(sum(x*x for x in q) / HEAD_DIM + EPSILON)
    scaled = [q[i] * reciprocal_norm * gamma[i] for i in range(HEAD_DIM)]
    output = [0.0] * HEAD_DIM
    theta_scale = BASE ** (-2.0 / HEAD_DIM)
    for pair in range(HEAD_DIM // 2):
        angle = positions[position_axis(pair)] * (theta_scale ** pair)
        c, s = math.cos(angle), math.sin(angle)
        x, y = scaled[pair], scaled[pair + HEAD_DIM // 2]
        output[pair] = x*c - y*s
        output[pair + HEAD_DIM // 2] = x*s + y*c
    return output


def fixture(positions=(3, 5, 11, 17)):
    from tools.generate_fixtures import synthetic_fixture
    result = synthetic_fixture(positions)
    result['expected_float64'] = [reference_head(row, result['gamma'], positions)
                                  for row in result['q']]
    return result
