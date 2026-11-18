# 多樣化資料重訓轉彎 BC（2026-11-17）

維持既有混合架構：直線、置中、重定位、轉彎准入、對齊確認與走道交接由 FSM 控制；只有 TURN_LEFT／TURN_MAIN 的非零原地旋轉角速度交給模型。原模型與實車服務沒有更換。不是完整路口雙動作學習，也不是 PPO。

本輪已完成：新版正常 4／4、受擾 4／4，舊版同情境受擾 4／4；接管與獨立出口核對通過。見 [完整結果](../logs/turn_bc/20261117_diverse/RESULTS.md)。尚未證明新版更可靠，保留原版作基準。

## 新示範如何接到原架構

從 `logs/rough_bc_demos/diverse_20261116_v2/` 的 31 個成功回合中，選取標籤有效、前後階段相同、純轉彎且尚未對齊的幀。沒有把失敗、停車、重定位與直線幀當成角速度答案。

以原始雷達重新使用現有 `turn_bc.encode()`，避免把 159 維輸入硬塞入 80 維模型。四類任務與局部中心／航向使用當時的因果量測，世界座標不進模型。原始完整回合的 train／validation／test 切分保持不變。

| 切分 | 成功來源回合 | 有效轉彎幀 |
|---|---:|---:|
| train | 23 | 312 |
| validation | 4 | 47 |
| test | 4 | 125 |

完整路口的 3,830 個有效標籤並非全部都是原地旋轉；本版實際使用 484 幀。完整雙動作資料仍保留，未刪除。

## 訓練與模型選擇

沿用 80→64→64→1 有界正增益模型：增益 0.25～4.0，乘上航向誤差，角速度上限 ±0.6 rad/s。重定位後原控制器仍限制為 ±0.25 rad/s，訓練損失也套用同樣裁切，避免把執行上限誤當成模型應自行學會的不同規律。

每個任務等權，再對該任務的完整回合等權，避免長恢復片段主導資料。三個固定訓練種子 0／1／2 各跑 600 epochs；只依驗證集平衡 MSE 選 checkpoint 與種子，測試集不參與選擇。沒有合併上一版資料來調整測試表現。

資料、權重、三種子紀錄與 provenance 保存於 `logs/turn_bc/20261117_diverse/`。`model/model.pt` 可由既有 `TurnPolicy` 載入，但本次只在獨立 Gazebo 評估中使用，沒有更改預設啟動模型。

## 閉迴路驗證方式

domain 47／partition `4wd_env_47` 的原物理碎石世界，每步推進 0.2 模擬秒，保留架高、斜坡與角柱。使用新 seed 基準 20261117，測試順序在開始前固定：新版四類正常轉彎，接著四類 18 cm 進場方向擾動，各比較舊版與新版，共 12 回合。這些起始抖動不同於訓練資料，但仍是相同地形與相似情境，不代表獨立地形泛化。

沿用原教師的安全器、重定位次數、速度上限與幾何條件。兩個模型都只替換教師的 turn_policy；恢復與直線仍是同一 FSM。成功需到達正確出口且停穩；另記模型接管、重試、定位恢復、安全阻擋與推論耗時。保存原始雷達、每步決策與模型呼叫；沒有接觸感測器，不能宣稱無碰撞。最後保留世界暫停。

閉迴路與離線誤差結果保存於同目錄 `RESULTS.md`。

## 重現

使用新的輸出目錄，避免覆寫既有實驗：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 rl_junction/train_diverse_turn_bc.py prepare \
  --source logs/rough_bc_demos/diverse_20261116_v2 --output logs/turn_bc/NEW_RUN/dataset
python3 rl_junction/train_diverse_turn_bc.py train \
  --source logs/turn_bc/NEW_RUN/dataset --output logs/turn_bc/NEW_RUN/model --epochs 600
```

Gazebo 已載入正確世界、scan／odom／cmd_vel 原橋接仍在且沒有其他速度發布器時，依 [ROUGH_DEMOS.md](ROUGH_DEMOS.md) 加上世界控制／姿態／時鐘橋接，再在相同 domain／partition 執行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export IGN_PARTITION=4wd_env_47 GZ_PARTITION=4wd_env_47
python3 rl_junction/evaluate_diverse_turn_bc.py \
  --model logs/turn_bc/NEW_RUN/model/model.pt \
  --baseline logs/turn_bc/20261116_pilot/model_v2/model.pt \
  --output logs/turn_bc/NEW_RUN/closed_loop
python3 scripts/audit_diverse_turn_bc.py logs/turn_bc/NEW_RUN/closed_loop
```

## 解讀限制

教師本身是有界航向回授，這個結構化 BC 主要模仿同一規律。低離線誤差或兩者都能完成，不能證明新模型更會補償碎石。擾動後恢復仍由 FSM 做，不能全部算成模型學會恢復。若新舊差異不明顯，保留原基準，下一階段應補獨立物理擾動與恢復資料、比較完整路口策略；不要直接以這個單角速度 actor 載入舊雙動作 PPO。
