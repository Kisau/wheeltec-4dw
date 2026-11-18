# 完整轉彎 BC 與修正示範

這是獨立對照程式，保留既有巡邏與原地旋轉 BC。只允許 Gazebo domain 47／partition `4wd_env_47`，沒有變更實車服務或既有模型。

本次已補 16 個成功修正示範（2,013 筆有效標籤），另保留 2 個停滯中止案例。完整 BC 第一版四種開發情境全數未完成，失敗紀錄保留；已加入政策失敗位置的教師重收與明確准入裁切，重訓第二版。**目前仍是未通過功能驗收的模擬對照模型，不替換原基準。** 本輪資料與結果位於 `logs/full_junction_bc/20261117/`。

第二版正常情境 0／4，旋轉 BC 對照 4／4。保留情境中又出現距中心約 21 cm 的近零速度平衡，已停止後續重複導航測試，保存中斷與未執行項目；完整批次尚未跑完。見 [實際結果與卡住原因](../logs/full_junction_bc/20261117/RESULTS.md)。

## 控制範圍

| 區段 | 動作來源 |
|---|---|
| 長距離主幹道、支線巡航 | 原 FSM |
| CENTER／RETURN_CENTER | 完整轉彎 BC 的線速度與角速度 |
| TURN_LEFT／TURN_MAIN | 完整轉彎 BC 的線速度與角速度 |
| ENTRY_LEFT／EXIT_MAIN | 完整轉彎 BC 的線速度與角速度 |
| 階段切換、無效幾何、重定位與障礙保護 | 原 FSM／外部安全器 |

模型在置中完成前已能控制 v/w；不是只替換旋轉公式。原 FSM 仍觀測幾何、更新階段、保留轉彎准入／中心偏移／重定位／出口確認條件。FSM 在背景計算的教師動作另作診斷，不作模型輸入，也不混稱為模型輸出。這是一個保留幾何守門條件的混合基準，並非取消所有局部狀態的端到端導航。

## 新修正示範

`collect_correction_demos.py` 在原物理碎石世界新增固定情境：

- 四種任務，剛開始轉彎時沿進場方向退移 18 cm，收集停止、重定位、重新置中及出彎。
- 四種任務，剛開始轉彎時橫移 6 cm，收集可恢復範圍內的修正。
- 四種任務，暫時抑制發布的動作 4 步（0.8 模擬秒），保留命令與實際執行不一致的序列；解除後觀察能否完成。
- 兩個獨立壓力案例，持續抑制 12 步，測試無進展看門狗能否停止。

姿態移位與軟體抑制明確記錄在事件中，不是摩擦／載重／實體打滑隨機化。抑制期間標籤無效但歷史保留。只將完整成功、停穩、出口核對通過的修正回合加入 train；失敗與舊壓力案例另存，不冒充成功教師動作。沒有以無效或失敗軌跡虛構正確修正答案。

原資料的 validation／test 完整回合保持原切分，新增修正資料只加入 train。來源程式與資產雜湊仍可核對；舊示範收集器未修改。

第一版實際失敗位置另由 `collect_failure_replays.py` 重收四個教師回合：選取模型已走偏、尚在可恢復角度內的位置，重新落地並讓原教師重新辨識、修正、完成出口。這是四次新物理 rollout，不是將失敗模型軌跡改標成功。來源失敗回合、步數與重置座標均保留。四回合全部成功；使用這些案例修正後，重複的開發情境不能再稱為獨立測試。

## 模型

8 幀因果歷史，每幀使用既有完整路口 159 維觀測；約 1.6 模擬秒。回合開頭重複第一幀補齊，視窗不跨回合，也不先刪除幾何失效／停車幀。有效遮罩只決定最後一幀是否計入損失。

MLP：`8×159 → 128 → 128 → 2`，tanh 有界輸出，再映射為 v≤0.4 m/s、|w|≤0.6 rad/s。平均與標準差只從 train 計算，任務／回合／局部階段平衡權重。三個訓練種子只按 validation 選 checkpoint，test 在選定後才評分。世界座標與 FSM 階段均不進策略輸入；階段只用於接管、採樣及安全。

## 外部安全

- 保留原雷達新鮮度、障礙淨空、幾何、中心偏離與轉彎重試條件。
- 需新鮮 `/odom`（與掃描差≤0.15 秒）；過期即停止輸出。
- 運行時依巡航上限裁切 v；重試時沿用 |w|≤0.25 rad/s。
- 修正版將置中時的原有速度包絡、朝向不適合平移時停止、向路口中心對正、准入區內停車確認及轉彎對齊確認，明確放在模型外部。這些裁切只刪除禁止的分量，不換成教師命令；每筆前後值與介入幀數均記錄。成績應解讀為模型與護欄的組合，不是完全無介入模型。
- 局部動作有持續需求但量測速度極低時，8 幀後以無進展看門狗中止；這是停滯指標，不是電機故障或碰撞診斷。
- 無效／非有限模型輸出立即停車，沒有偷偷改用教師繼續完成。
- 保留人工 STOP 優先權與原有模式重設流程。

## 檔案與重現

- `rl_junction/full_junction_bc.py`：資料、歷史、雙動作模型與訓練。
- `rl_pig_pen/full_junction_controller.py`：局部接管與外部安全。
- `rl_pig_pen/control_modes_right_then_left_full_bc.py`：獨立 ROS 模擬入口。
- `rl_junction/evaluate_full_junction_bc.py`：固定情境的新舊比較、擾動與停滯測試。
- `scripts/audit_full_junction_bc.py`：獨立資料與閉迴路核對；重算模型輸出確認接管紀錄。

原 Gazebo 與 scan／odom／cmd_vel 橋接需已就緒，世界控制橋接見 [ROUGH_DEMOS.md](ROUGH_DEMOS.md)。不要同時啟動其他速度發布器。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export IGN_PARTITION=4wd_env_47 GZ_PARTITION=4wd_env_47
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 rl_junction/collect_correction_demos.py --output logs/rough_bc_demos/NEW_CORRECTIONS
python3 scripts/audit_full_junction_bc.py corrections logs/rough_bc_demos/NEW_CORRECTIONS
# 可選：從已完成的開發失敗紀錄重收教師修正，另存新目錄
python3 rl_junction/collect_failure_replays.py \
  --source logs/full_junction_bc/FAILED_DEVELOPMENT \
  --output logs/rough_bc_demos/NEW_REPLAYS
python3 rl_junction/full_junction_bc.py \
  --source logs/rough_bc_demos/diverse_20261116_v2 \
  --source logs/rough_bc_demos/NEW_CORRECTIONS \
  --source logs/rough_bc_demos/NEW_REPLAYS \
  --output logs/full_junction_bc/NEW_RUN/model
python3 rl_junction/evaluate_full_junction_bc.py \
  --model logs/full_junction_bc/NEW_RUN/model/model.pt \
  --turn-baseline logs/turn_bc/20261117_diverse/model/model.pt \
  --output logs/full_junction_bc/NEW_RUN/evaluation
python3 scripts/audit_full_junction_bc.py evaluation logs/full_junction_bc/NEW_RUN/evaluation
```

預設完整驗證：四類正常情境各比較完整 BC 與旋轉 BC，完整 BC 再跑四類未見進場方向擾動及兩個持續停滯情境。失敗照常保留，核對通過不等於導航成功。單一固定物理碎石地形，沒有接觸感測器；不能宣稱實車泛化或零碰撞。未做 PPO／Transformer 訓練。收集與評估結束均送零速度並保留世界暫停。
