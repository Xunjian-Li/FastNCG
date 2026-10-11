"""Compare same-fixture MATLAB / Julia output; requires only Python stdlib."""
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
for case in ('stationary', 'quadratic', 'rosenbrock', 'negative_curvature'):
    def read(suffix):
        with (root / (case + suffix)).open() as f:
            return [[float(v) for v in row] for row in csv.reader(f) if row]
    a, b = read('_julia.csv')[0], read('_matlab.csv')[0]
    assert a[:4] == b[:4], (case, 'oracle/iteration counts', a, b)
    assert max(abs(x-y)/(1+max(abs(x),abs(y))) for x,y in zip(a[4:],b[4:])) < 1e-12
    ah, bh = read('_julia_history.csv'), read('_matlab_history.csv')
    assert len(ah) == len(bh), case
    error = 0.0
    for x, y in zip(ah,bh):
        assert x[2:] == y[2:], (case, 'step / M / CG iterations / direction type')
        error = max(error, *(abs(v-w)/(1+max(abs(v),abs(w))) for v,w in zip(x[:2],y[:2])))
    assert error < 1e-12, (case,error)
    print(f'{case}: iter={int(a[0])}, HVP={int(a[3])}, counters/branches match, scaled history error={error:.3e}')
