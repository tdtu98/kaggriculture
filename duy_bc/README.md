# Duy behavior-cloning handoffs

[`00_codex_baseline/`](00_codex_baseline/) is Duy's current reproducible
100-win behavior-cloning baseline for sharing with Tu.

[`BC_EVALUATION_GUIDE.md`](BC_EVALUATION_GUIDE.md) explains why the project
uses sequence-aware cloning metrics, how to run validation, and how to evaluate
a new model architecture with the same metric implementation.

Replay data, prepared shards, checkpoints, run outputs, caches, and virtual
environments are not included. Each collaborator supplies their own replay
corpus by editing the baseline configuration.

The root-level `duy_rl/` project is intentionally separate and remains
available for Duy's later experiments.
