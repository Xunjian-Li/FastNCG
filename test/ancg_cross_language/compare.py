"""Compare saved MATLAB/Julia ANCG histories without assuming bitwise parity."""
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read(path):
    with path.open() as f:
        return [[float(x) for x in row] for row in csv.reader(f) if row]

def scaled_error(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return math.inf
    return max((abs(x-y)/(1+max(abs(x),abs(y)))
                for ar, br in zip(a,b) for x,y in zip(ar,br)), default=0.0)

lines = ["# ANCG MATLAB–Julia comparison", "",
    "MATLAB: " + (root/'matlab_version.txt').read_text(), "",
    "Pinned upstream: `45532f314fbf2f0c064f676c64bb9e53db0c6ee6`.", "",
    "Both executions use the same CSV inputs, explicit analytic Hessians, one BLAS thread, "
    "and a gradient tolerance of `1e-7`. Histories are not forced to agree.", "",
    "| Case | Iterations Julia / MATLAB | CG-HVP Julia / MATLAB | Same counters | Same step types and lengths | Maximum scaled history difference |",
    "|---|---:|---:|---|---|---:|"]
details=[]
for name in ('quadratic','fail_sol','fail_nc','rosenbrock','polytope','tregression','logsumexp'):
    folder=root/name
    jc=read(folder/'julia_counts.csv'); mc=read(folder/'matlab_counts.csv')
    jh=read(folder/'julia_history.csv'); mh=read(folder/'matlab_history.csv')
    js=read(folder/'julia_steps.csv'); ms=read(folder/'matlab_steps.csv')
    jx=read(folder/'julia_x.csv'); mx=read(folder/'matlab_x.csv')
    err=scaled_error(jh,mh)
    error_label=f'{err:.3e}' if math.isfinite(err) else 'different lengths'
    lines.append(f'| {name} | {int(jc[0][0])} / {int(mc[0][0])} | '
                 f'{int(jc[4][0])} / {int(mc[4][0])} | {jc==mc} | {js==ms} | {error_label} |')
    stopj=(folder/'julia_stop.txt').read_text(); stopm=(folder/'matlab_stop.txt').read_text()
    details.append(f'- **{name}**: final loss {jh[-1][0]:.17g} / {mh[-1][0]:.17g}; '
                   f'gradient norm {jh[-1][1]:.6e} / {mh[-1][1]:.6e}; '
                   f'maximum scaled final-x difference {scaled_error(jx,mx):.3e}; '
                   f'stop reasons agree: {stopj==stopm}.')
    # These analytic fixtures have deterministic branch/counter expectations.
    if name in ('quadratic','fail_sol','fail_nc'):
        assert jc == mc and js == ms and stopj == stopm
        assert err < 1e-13 and scaled_error(jx,mx) < 1e-13
    else:
        assert stopj == stopm == 'First-order tolerance satisfied.'
        assert jh[-1][1] <= 1e-7 and mh[-1][1] <= 1e-7

assert read(root/'matlab_capcg_2.csv') == [[-0.25,1.0,0.125,5.0,1.0]]
assert read(root/'matlab_capcg_-4.csv') == [[0.5,2.0,-1.0,5.0,1.0]]
warm=(root/'matlab_warmstart.txt').read_text()
assert warm != 'unexpected success'
lines += ['', '## Final values', '', 'Values below are Julia / MATLAB.', '', *details, '',
          'The one-dimensional positive/negative-curvature CappedCG fixtures agree exactly, '
          'including five HVPs and continued execution after the initial negative-curvature test.', '',
          'The upstream warm-start failure was reproduced in MATLAB: `' + warm.replace('\n',' ') + '`.', '',
          'Scaled differences use `abs(a-b)/(1+max(abs(a),abs(b)))`. '
          'Matching final convergence is weaker than matching the full trajectory; '
          'the table reports both rather than treating convergence as exact parity.', '']
report='\n'.join(lines)
print(report)
if len(sys.argv)>2:
    Path(sys.argv[2]).write_text(report)
