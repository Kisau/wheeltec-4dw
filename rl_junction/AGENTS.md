# Training records

User preference (2026-11-18): all future training must write TensorBoard records.

- For BC, PPO, residual PPO, Transformer, and other future training runs, enable TensorBoard before training begins. Keep CSV/JSON and checkpoints too.
- Record training/validation losses, reward, episode length, available success/intervention metrics, and run configuration using meaningful step/epoch axes. Flush records on completion and interruption.
- PPO entry points use an SB3 logger with the tensorboard output format. For custom BC entry points whose source hashes are part of existing checkpoint contracts, use `train_with_tensorboard.py` to mirror their live metrics without modifying the contracted trainer.
- Historical logs imported later must be labeled `historical_import`; do not invent missing metrics or imply that they were recorded live. Keep validation/test separate from training.
- Preserve existing model weights and their contracts. Use a new output directory for each new training run.
