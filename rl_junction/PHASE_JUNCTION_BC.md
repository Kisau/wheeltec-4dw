# 有階段條件的完整轉彎 BC

獨立實驗版本；保留先前的原地旋轉 BC、不帶階段的完整 BC、權重與失敗紀錄。沒有更換 Xavier 服務或預設模型，入口限制 Gazebo domain 47／partition `4wd_env_47`。

## 本次修正

| 已發現問題 | 實作 |
|---|---|
| 尚未置中就朝出口旋轉 | 明確提供目前六種局部階段；置中階段的目標航向改為進場軸，旋轉／出彎階段才指向出口 |
| 出彎偏離 | 增加當前出口走道的角度、兩側偏差及有效旗標；學習路口／走道轉向融合；牆面尚未連續穩定時限制 0.07 m/s，穩定後才允許至 0.12 m/s |
| 模型在中心前近零輸出 | 依局部幾何誤差判斷進展，不只看命令大小；連續 2 模擬秒未改善時先停車，再交由原 FSM 作有限修正 |

長距離直線仍由 FSM 控制。模型控制 CENTER／RETURN_CENTER、TURN_LEFT／TURN_MAIN、ENTRY_LEFT／EXIT_MAIN 的局部 v/w；FSM 仍決定階段切換、准入、確認及安全。這是有監督階段條件的混合模型，不是模型自行推斷所有階段的端到端策略。

## 觀測與模型

8 幀因果歷史，每幀 168 維：

- 原本 159 維雷達、任務、局部中心、目標角、量測速度、前一執行動作與 dt。
- 6 維目前局部階段 one-hot。
- 3 維當前走道角度、兩側距離差與有效旗標。

階段在當前掃描處理後、模型輸出前即可由主管取得，不是未來階段或正確答案。訓練使用同一時點的 next_phases 欄位，階段切換幀仍不作動作監督；完整歷史保留，沒有跨回合拼接。世界座標、教師動作與未來觀測不進模型。

模型以 128／128 MLP 學習有界回授增益、出彎速度與路口／走道融合。置中線速度依目前中心誤差作回授；原地旋轉的方向由目標角誤差決定、增益保持正值；出彎由有界速度與方向／偏差回授控制。這是結構化 BC，並非完全自由的雙動作回歸。沒有用固定最低速度強推車子；模型／護欄無法完成時應停止或啟動明確修正。

因此這版同時改變了階段輸入、走道量測與動作結構，成績改善不能全歸因於多一個階段標籤。採用相同原始資料與完整 episode 切分：39 train／4 validation／4 test，局部有效幀 4,576／474／649。正規化只由 train 計算，三種子各 400 epochs，僅依 validation 選模型。舊權重未覆寫。

## 幾何進展與安全介入

- 置中看目前距離，旋轉看目標角誤差，幾何出彎階段看剩餘入口深度。
- 2 模擬秒內至少改善 1.5 cm（位置）或 0.03 rad（旋轉）；達不到時先輸出零速度。
- 下一幀開始可交給原安全 FSM，標記 `FSM_CORRECTION`，每次最多 8 模擬秒、每個局部任務最多 2 次；失敗則停止。
- 階段切換時再停一幀並交回模型；記錄觸發事件、修正幀數及結果。不得把 FSM 救回包裝為模型自行完成。
- 幾何／里程計失效、障礙保護、速度、中心偏離、轉彎重試與人工 STOP 仍保持在模型外。沒有拿無效量測作主動修正依據。
- `projection_frames` 記錄模型要求被速度／准入／確認護欄裁切的幀；`fsm_correction_frames` 記錄進展檢查交回 FSM 的幀；`unassisted` 另看安全介入，不能只看成功率。
- WALL_CONFIRM 已不含完整路口幾何時，不虛構入口深度；使用原走道確認／超時保護。位置／角進展門檻是模擬起始參數，尚未實車標定。

## 檔案

- `phase_junction_bc.py`：階段／走道觀測、結構化模型與離線訓練。
- `../rl_pig_pen/phase_junction_controller.py`：模型接管、低速出彎與有界 FSM 修正。
- `../rl_pig_pen/control_modes_right_then_left_phase_bc.py`：獨立 ROS 模擬入口，參數 `phase_bc_model`。
- `evaluate_phase_junction_bc.py`：正常、保留擾動及停滯測試。
- `../scripts/audit_phase_junction_bc.py`：重算模型輸出、核對階段、出口、停車及修正來源。
- `../scripts/probe_phase_progress_recovery.py`：刻意注入零輸出，驗證 21 cm 卡點的停車／FSM 修正。這是護欄測試，不算模型導航成功。

資料與結果保存於 [20261117 實測報告](../logs/phase_junction_bc/20261117/RESULTS.md)。正常開發案例 4/4；保留導航案例 3/4，成功的三例皆有 FSM 修正；另一例在擾動施加前幾何定位逾時。兩例持續動作抑制皆觸發停車保護，零輸出卡點也驗證先停車再由 FSM 修正。尚未完整解決左轉進入的定位問題。

## 重現

以下使用新目錄，保留現有實驗。Gazebo 與必要橋接需先就緒，且沒有其他速度發布器；橋接方式同 [ROUGH_DEMOS.md](ROUGH_DEMOS.md)。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 rl_junction/train_with_tensorboard.py --script phase_junction_bc.py -- \
  --source logs/rough_bc_demos/diverse_20261116_v2 \
  --source logs/rough_bc_demos/corrections_20261117 \
  --source logs/rough_bc_demos/policy_replays_20261117 \
  --output logs/phase_junction_bc/NEW_RUN/model --epochs 400

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export IGN_PARTITION=4wd_env_47 GZ_PARTITION=4wd_env_47
python3 rl_junction/evaluate_phase_junction_bc.py \
  --model logs/phase_junction_bc/NEW_RUN/model/model.pt \
  --turn-baseline logs/turn_bc/20261117_diverse/model/model.pt \
  --stage development --output logs/phase_junction_bc/NEW_RUN/development
python3 scripts/audit_phase_junction_bc.py evaluation logs/phase_junction_bc/NEW_RUN/development
```

正常情境通過後，使用 `--stage heldout` 另存新目錄，執行四種保留擾動與兩個持續動作抑制案例。預期停滯保護中止不算導航成功；擾動若未施加也不能宣稱恢復通過。固定物理碎石地形、沒有接觸感測器，未證明實車泛化或零碰撞；不自動部署或啟動 PPO。
