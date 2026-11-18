# TensorBoard 與接續訓練

使用者要求：之後所有訓練都記錄 TensorBoard。專案根目錄及本目錄的 AGENTS.md 已保留這個偏好。

## 查看目前資料

本機 `http://127.0.0.1:6006` 已提供這輪歷史匯入的資料；選擇名稱含 `historical_import/ppo_512` 的 run。來源為原 console、Monitor CSV 與 checkpoint，不是當時即時記錄。512 步只有三個完整回合；最後一批 optimizer 更新沒有原始 loss 日誌，未補造。

圖表與 CSV 在 `logs/residual_phase/20261117/training_curves/`。啟動自己的總覽可用 `python3 -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006 --load_fast=false`；同一 port 只啟動一個 server。

## 新的 BC

不改動舊訓練器的來源雜湊契約，改以 live mirror 入口啟動：

```bash
python3 rl_junction/train_with_tensorboard.py --script phase_junction_bc.py -- \
  --source logs/residual_phase/20261117/clean_base \
  --source logs/residual_phase/20261117/clean_fsm \
  --output logs/residual_phase/NEW_RUN/bc --epochs 400
```

每種子產生 `seed_N/tensorboard/`。按原 trainer 的 metrics 輸出頻率記錄 epoch、train／validation MSE；不虛構未輸出的 epoch 指標。此 wrapper 也支援用 pathlib 寫出的 metrics.csv。

## PPO 與接續

`train_residual_phase.py`、`train.py`、`continuous_train.py` 的新訓練會保留 TensorBoard、CSV 與 JSON。訓練入口未加 `--resume` 代表新建 PPO；BC 仍讀取指定權重。

接續 residual PPO 範例（以下是操作說明，尚未啟動新一輪長訓練）：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export IGN_PARTITION=4wd_env_47 GZ_PARTITION=4wd_env_47
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 rl_junction/train_residual_phase.py \
  --bc logs/residual_phase/20261117/bc/model.pt \
  --resume logs/residual_phase/20261117/ppo/residual_ppo.zip \
  --output logs/residual_phase/20261118/continued_01 \
  --steps 1024 --seed 18
```

需先備妥相同隔離 Gazebo 與橋接，且沒有其他速度發布器、不要按 GUI play。此指令會載入 PPO policy／value 與 optimizer 狀態，凍結同一個 BC，以新回合收集額外 1024 步，總步數由 512 累計到 1536。新 run 的 `metrics/` 會寫 TensorBoard；沿用全域步數，不覆寫舊 run。

物理模擬、回合歷史與隨機數從新回合開始，因此是模型與 optimizer 的接續，不是逐位元重播中斷當下。不同 BC、不同 residual 上限、metadata 不符或重用輸出目錄會拒絕。

下一輪模型仍應經獨立 validation 選擇與未見測試比較；目前接續入口不自動宣稱訓練已收斂，也不自動部署或取消安全檢查。
