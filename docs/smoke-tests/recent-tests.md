## Summary

- Align Isaac Lab 3.0/Python 3.12 runtimes while preserving the image’s matched Torch/CUDA stack and verifying installed dependency versions.
- Harden Viewer dataset-contract validation, launcher behavior, annotation workflows, and accessibility.
- Reconcile Copilot guidance and documentation, replacing environment-specific examples with placeholders.
- Merge upstream main at d435c7d3 without rewriting fork history.

## Validation

- Automated headless SKRL/PPO smoke passed on one RTX 4090 using the pinned Isaac Lab 3.0.0-beta2-post1 image.
- Ran Isaac-Velocity-Rough-Anymal-C-v0 on CUDA: 4 environments, 5 iterations / 120 vectorized steps, seed 42. Training completed and the container exited with code 0.
- Saved 5 checkpoints; reloaded the final checkpoint and verified all 58 contained tensors were finite.
- CUDA matrix operations, convolution, and native torchvision CUDA NMS passed. All 123 installed workflow dependency versions matched the frozen lock.
- Passed 188 targeted training tests, 152 CI contract tests, and 89 PowerShell regression tests. Bootstrap regressions, workflow validation, frontend type-check, and production build also passed.

This is a short single-GPU smoke, not validation of convergence, resumed training, camera rendering, or Azure/OSMO execution. The beta image still emits setup/OmniHub warnings, and its built-in Kit service healthcheck did not pass during the short batch job; the explicit training and GPU checks passed.
