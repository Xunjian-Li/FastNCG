# Investigation of differing CappedCG counts

These observations supplement [RESULTS.md](RESULTS.md). They are diagnostics
of the recorded runs, not a claim of bitwise or formal equivalence.

For each selected call, a tracing wrapper saved the incoming Hessian, gradient,
regularization parameter, and tolerance to 17-digit CSV files. Both the pinned
MATLAB `capCG` and Julia `ANCGReference.capcg` then consumed each saved input.
The wrappers did not change solver arithmetic. Both runtimes used one BLAS thread.

| Selected call | Saved input origin | Julia CG iterations / HVP | MATLAB CG iterations / HVP | Maximum absolute direction difference |
|---|---|---:|---:|---:|
| Student-t outer 5 | Julia | 42 / 128 | 42 / 128 | 4.928e-7 |
| Student-t outer 5 | MATLAB | 43 / 131 | 43 / 131 | 4.306e-7 |
| LogSumExp outer 215 | Julia | 2 / 8 | 2 / 8 | 3.469e-18 |
| LogSumExp outer 215 | MATLAB | 3 / 11 | 3 / 11 | 2.683e-18 |

All four replays returned direction type 1 (SOL). The iteration and HVP counts
follow the input origin in these cases, rather than the execution language.
This supports sensitivity to accumulated floating-point differences as the
explanation for these particular count discrepancies. Directions are still
not bitwise identical, especially after the longer Student-t CG solve.

Student-t first differs in CG iteration count at outer iteration 5. Before this
call, the two Hessians differ by at most approximately 3.569e-10 in absolute
value and the gradients by 2.552e-11. The regularization inputs are
53.583925115065504 (Julia) and 53.58392511506986 (MATLAB).
Near convergence the objective comparisons also diverge: at outer iteration 14,
Julia accepts a full step and converges, while MATLAB accepts a step of 0.015625
and continues. The complete solves finish in 14 and 16 iterations respectively,
with final objective values differing by approximately 3e-14.

LogSumExp first differs in CG iteration count at outer iteration 215. Its
regularization inputs are 0.07052390040660986 (Julia) and
0.07052369127757842 (MATLAB). Both complete solves take 622 outer iterations
with identical recorded step lengths and direction types, but their CG-HVP
totals differ by 24.

No acceptance tolerance, CappedCG stopping condition, or arithmetic expression
was changed to force agreement. Full-run input generation and comparisons are
reproducible using [README.md](README.md); the selected-call tracing and replay
above were additional diagnostic experiments, not assertions in that harness.
