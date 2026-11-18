# 使用者的訓練紀錄要求

使用者於 2026-11-18 要求：之後所有模型訓練都必須使用 TensorBoard 記錄。

- 新的 BC、PPO、Transformer 等訓練，啟動前就接上即時 TensorBoard，另保留 CSV/JSON 與 checkpoint。
- 訓練與驗證分開，正確標示步數／epoch；結束或中斷時 flush。事後匯入舊日誌標為 historical_import，不補造缺失數值。
- 不為了加入紀錄破壞既有模型的來源雜湊契約。自訂 BC 可使用 `rl_junction/train_with_tensorboard.py`，PPO 使用包含 tensorboard 的 SB3 logger。
- 詳細路口訓練指引見 `rl_junction/AGENTS.md`。
