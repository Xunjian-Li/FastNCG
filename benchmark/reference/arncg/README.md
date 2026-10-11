# ARNCG reference and benchmark configuration

Source: https://github.com/miskcoo/ARNCG

Pinned commit: `8605f1e13386f3d0908a3d4a6dce9ce555d890bf`.
The three MATLAB files are unmodified; the upstream MIT license is included.

SHA-256:

- `AdapNewtonCG.m`: `d6acf3a0bf1f4c86c3558913b1572f9e84677bff86cb98cbbafd98383b68d5f5`
- `CappedCG.m`: `6abe1a71ff1d3b42243f3c10be62310a92b7621ddc2f43988796986f4ae4d8f7`
- `TestCUTEst.m`: `505e1675a75e36a37fbc8fde829926c704d4b8b26a5889da1cb33d429dac90f5`

`arncg_official_optimize` in `src/ARNCG.jl` translates the Euclidean,
HVP-based, `cg_policy='recompute'` implementation. Defaults correspond to
`TestCUTEst.m` index 2: ARNCG_g, theta=1, lambda=0 (fallback disabled). This is
the authors' experimental variant, not the full theoretical fallback algorithm.
The benchmark overrides the gradient tolerance to 1e-7 for all methods.

Preserved source behavior includes the CG iteration cap returning SOL, the
historical search's `if it > 0` condition and recurrence state, and ordinary
objective subtraction. If the historical search cannot produce a direction,
MATLAB fails because output Z is unset; Julia now raises `UndefVarError(:Z)`.
It no longer invents a direction and passes ERR to the negative-curvature path.

Benchmark iteration counts are the number of executed outer steps (the length
of MATLAB `records`), not the loop index after checking convergence. An initially
stationary point therefore has zero iterations. HVP counts count actual calls,
including initial CG products and the negative-curvature scaling product, rather
than only summing CG iteration indices. These are reporting adaptations.
`details` exposes pre-step loss/gradient, alpha, M, CG-iteration and direction-type
histories, plus the authors' Hessian-evaluation counter. Verbose MATLAB formatting
and per-record timing are not reproduced; runtime is Julia wall-clock time.

See [the executable cross-language checks](../../../test/arncg_cross_language/README.md).
Agreement on those fixtures is not proof of identical trajectories on all inputs.
The rare historical-search failure path has been checked against source but was
not reached by those numerical fixtures.
