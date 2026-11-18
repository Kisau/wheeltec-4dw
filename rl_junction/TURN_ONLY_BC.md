# 右先左後：轉彎 BC、直線 FSM

獨立入口 `rl_pig_pen/control_modes_right_then_left_bc.py`，不會將既有入口自動改成 BC，也不會改 Xavier 的服務。

## 誰控制什麼

| 工作 | 控制者 |
|---|---|
| 主幹道前進／倒退、支線前進／倒退 | 原 RightThenLeftPatrolController |
| 路口偵測、置中、轉彎准入 | 原 FSM 與局部幾何 |
| TURN_LEFT／TURN_MAIN 原地轉彎的非零角速度 | BC 模型 |
| 車頭對齊確認、進入走道、交回直線 | 原 FSM |
| 雷達失效、障礙、偏離中心、外部停車、速度上限 | 原保護；模型不能解除 |

模型只輸出一個角速度，線速度固定為零。不是讓模型接管完整路口連續進出；之後如擴大範圍，需另立資料與模型契約。舊巡邏程式仍保留，新的右先左後控制器在沒有指定 turn_policy 時沿用原本轉彎公式。

## 第一版資料與模型

`logs/turn_bc/20261116_pilot/`：

- `dataset_v2/dataset.json`：目前資料的來源、世界／mesh／教師雜湊、逐段資料與切分。
- `model_v2/model.pt`、`model_v2/model.json`：目前模型與契約、離線誤差。
- `model_v2/training_provenance.json`：資料 manifest、模型與訓練日誌雜湊。
- `closed_loop_v2/`：目前的獨立位置核對、原始雷達與混合控制日誌。
- `status.json`：目前階段與失敗版／修正版的對應位置。

初始示範取自 2026-11-16 已完成的實體碎石巡邏。四類任務（右轉入、右返回、左轉入、左返回）各三段；共 356 個有效轉彎幀。每類第一段 train、第二段 validation、第三段 test；不同段沒有混幀，但仍來自同一輪、同一固定地形，不能代表獨立地形或實車泛化。

輸入 80 維：72 個 LiDAR 最小距離分箱、四類任務 one-hot、局部路口中心 x/y、目標航向誤差 sin/cos。沒有影像、世界座標或未來幀。v2 使用 80→64→64→1 MLP 學習 0.25～4.0 的正回授增益，乘上量測航向誤差並限制在 ±0.6 rad/s；對齊完成仍由原本測量與連續確認判定。這是有結構約束的 BC，而非無限制的端到端角速度回歸。

v1 的無限制 MLP 在第一次閉迴路測試停在約 0.17 rad 航向誤差、輸出接近零，未完成第一個轉彎；已中止，保留於 `closed_loop/` 與 `model_unconstrained_v1/`。v2 的方向與正增益約束針對這個零速度平衡問題，不放寬原 FSM 的對齊門檻；新資料、權重及結果使用 `dataset_v2/`、`model_v2/`、`closed_loop_v2/`。

目前教師本身就是有界的航向回授，v2 的低離線誤差主要表示它學會模仿該回授規律，不代表已學會比教師更好的碎石補償或打滑恢復。這需要新擾動／恢復示範與獨立實驗支持。

資料排除轉彎狀態切入幀（其 heading 仍是置中目標）、失效幾何、外部停車與零命令；保留來源負載事件紀錄。模型版本與觀測／物理資產雜湊綁定，不能靠修改 metadata 載入舊模型。新增 hook 是明確的教師來源轉接，原控制器雜湊保存在資料來源 manifest。

## 重現

先收集／離線訓練，請勿與不限執行緒的數值測試同時佔用 Gazebo CPU：

以下是重現命令；新實驗請改用新的時間戳目錄，保留現有資料與評估依據。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 rl_junction/turn_bc.py collect \
  --source logs/env_47/retests/20261116_131808 \
  --output logs/turn_bc/20261116_pilot/dataset_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 rl_junction/turn_bc.py train \
  --source logs/turn_bc/20261116_pilot/dataset_v2 \
  --output logs/turn_bc/20261116_pilot/model_v2 --epochs 1000
```

Gazebo domain 47 已啟動、車子位於起點、且沒有其他速度發布器時：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 rl_pig_pen/control_modes_right_then_left_bc.py --ros-args \
  -p initial_mode:=AUTO -p use_sim_time:=false \
  -p cruise_speed:=0.4 -p max_linear:=0.4 \
  -p bc_model:="$PWD/logs/turn_bc/20261116_pilot/model_v2/model.pt"
```

不要與原控制入口並跑。`closed_loop` 的 geometry 紀錄中 `controller.command_owner=BC_TURN` 表示該筆角速度來自模型；其他階段保留原 FSM。重設 AUTO 會經 controller factory 重新建立帶有相同模型的控制器，避免重設後悄悄退回教師。

這一版是小規模 BC 起始實驗。下一階段應收更多完整回合、偏心與姿態擾動及恢復示範；擴充資料收集器為多來源／多地形分組切分，再決定是否做 PPO。當前純 PyTorch 單角速度模型與舊 SB3 雙動作模型不相容，不能直接交給舊 PPO 腳本載入。

## 2026-11-16 完成結果

修正版 `model_v2/model.pt` 完成一輪 Gazebo 閉迴路：右三條→左三條→起點端停止。六個端點的世界座標與順序均通過；約 25 分 48 秒，最後 v/w 為零，最後 0.8 秒世界位置最大變化約 0.000000046 m。停止位置距原設定起點約 20.7 cm，並非精準回到原座標。

`closed_loop_v2/ownership_audit.json` 核對十二段轉彎全部有模型輸出，共 346 筆 BC 命令；沒有非零轉彎命令漏掉 BC 標記，也沒有 BC 在非轉彎階段接管或輸出線速度。來源、模型與物理網格雜湊均保持一致。

離線測試：train 123 幀、validation 118 幀、test 115 幀；保留測試集角速度 MAE 約 0.000400 rad/s。這些幀來自十二段示範，不是 356 次獨立轉彎，也不是獨立地形測試。

本輪沒有 ERROR 或定位恢復；有啟動暫停及一次運行中的短暫雷達間隔，均恢復後完成。模型與保護邏輯的七項單元測試通過，涵蓋模型接管、預設原控制器、非有限輸出停車、中心偏離／外部停車阻擋，以及正增益方向／界限與可訓練梯度。Gazebo 保留；測試控制節點已停止並完整關閉壓縮日誌。

這是單一固定地形的起始驗證；沒有接觸感測器，不能宣稱零實體接觸。尚未部署實車，也尚未證明擾動／打滑恢復能力。

同日再次重測：`logs/turn_bc/retests/20261116_150304/`。同一模型與參數完成右三條、左三條及起點端停止，約 25 分 32 秒；六個端點與十二段 BC 轉彎全部通過。另有安全設定快照與接管範圍比較說明。可用 `scripts/audit_turn_bc_retest.py --run <紀錄目錄> --baseline <上次目錄>` 在測試節點退出後重做唯讀核對。
