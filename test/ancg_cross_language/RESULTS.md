# ANCG MATLAB–Julia comparison

MATLAB: 26.1.0.3203278 (R2026a)

Pinned upstream: `45532f314fbf2f0c064f676c64bb9e53db0c6ee6`.

Both executions use the same CSV inputs, explicit analytic Hessians, one BLAS thread, and a gradient tolerance of `1e-7`. Histories are not forced to agree.

| Case | Iterations Julia / MATLAB | CG-HVP Julia / MATLAB | Same counters | Same step types and lengths | Maximum scaled history difference |
|---|---:|---:|---|---|---:|
| quadratic | 10 / 10 | 50 / 50 | True | True | 0.000e+00 |
| fail_sol | 1 / 1 | 5 / 5 | True | True | 0.000e+00 |
| fail_nc | 1 / 1 | 5 / 5 | True | True | 0.000e+00 |
| rosenbrock | 185 / 185 | 10222 / 10222 | True | True | 6.005e-13 |
| polytope | 746 / 746 | 5575 / 5575 | True | True | 8.433e-16 |
| tregression | 14 / 16 | 1738 / 2324 | False | False | different lengths |
| logsumexp | 622 / 622 | 16217 / 16241 | False | True | 2.935e-08 |

## Final values

Values below are Julia / MATLAB.

- **quadratic**: final loss 8.1674258600864146e-16 / 8.1674258600864146e-16; gradient norm 4.041640e-08 / 4.041640e-08; maximum scaled final-x difference 0.000e+00; stop reasons agree: True.
- **fail_sol**: final loss 0 / 0; gradient norm 1.000000e+00 / 1.000000e+00; maximum scaled final-x difference 0.000e+00; stop reasons agree: True.
- **fail_nc**: final loss 0 / 0; gradient norm 1.000000e+00 / 1.000000e+00; maximum scaled final-x difference 0.000e+00; stop reasons agree: True.
- **rosenbrock**: final loss 3.614705103118408e-19 / 3.614705959395736e-19; gradient norm 6.009577e-10 / 6.009555e-10; maximum scaled final-x difference 5.551e-17; stop reasons agree: True.
- **polytope**: final loss 1.5636196943110586e-09 / 1.5636196943110607e-09; gradient norm 9.952608e-08 / 9.952608e-08; maximum scaled final-x difference 1.982e-15; stop reasons agree: True.
- **tregression**: final loss 153.59013732257222 / 153.59013732257219; gradient norm 1.097271e-10 / 7.591845e-11; maximum scaled final-x difference 2.223e-13; stop reasons agree: True.
- **logsumexp**: final loss 1.4623357840359086 / 1.4623357840359044; gradient norm 9.394046e-08 / 9.392693e-08; maximum scaled final-x difference 7.422e-09; stop reasons agree: True.

The one-dimensional positive/negative-curvature CappedCG fixtures agree exactly, including five HVPs and continued execution after the initial negative-curvature test.

The upstream warm-start failure was reproduced in MATLAB: `Unrecognized function or variable 'j'.`.

Scaled differences use `abs(a-b)/(1+max(abs(a),abs(b)))`. Matching final convergence is weaker than matching the full trajectory; the table reports both rather than treating convergence as exact parity.
