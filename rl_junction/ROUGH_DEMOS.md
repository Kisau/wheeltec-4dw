# 多樣化完整路口 BC 示範

這批資料供下一版「直線 FSM、完整路口模型」使用，與目前 80 維、單角速度的 BC 模型分開；不會自動訓練或覆蓋現有權重。

本次收集已完成：32 回合中 31 個成功、3,830 筆有效監督，另保留獨立壓力案例。見 [實際結果與限制](ROUGH_DEMOS_RESULTS_20261116.md)。

## 實體模擬與案例

使用 Gazebo domain 47、partition `4wd_env_47` 的實體碎石世界，保留 6 cm 架高、15 cm 斜坡與槽鋼角柱。每次以 0.2 模擬秒步進，等待新鮮且對齊的雷達、里程計與世界姿態。世界姿態使用帶原始時間戳的 Gazebo 訊息；TF 轉換未保留該時間戳，因此不以 TF 當同步依據。

四類任務：

| 任務 | 進場 | 完成條件 |
|---|---|---|
| RIGHT_ENTER | 主幹道前進 | 轉入右支線，交回走道控制 |
| RIGHT_RETURN | 右支線倒退返回 | 轉回主幹道，繼續前進 |
| LEFT_ENTER | 主幹道倒退 | 轉入左支線，交回走道控制 |
| LEFT_RETURN | 左支線倒退返回 | 轉回主幹道，繼續倒退 |

每類預先規劃八個完整回合，合計 32 個：正常、左右約 4 cm 偏心、正負約 12° 朝向差、16 cm 進退方向受擾恢復、保留驗證的 8 cm／8° 混合偏差，以及保留測試的 18 cm 受擾恢復。另加固定種子的小幅起始抖動。分布涵蓋三個路口位置、0.2／0.3／0.4 m/s 巡航上限。

受擾恢復是在進入 TURN 時，沿原進場方向反向移位，讓原教師先停轉、重新置中，再完成路口；這是**離散姿態擾動**，不是摩擦或鬆石造成的真實打滑模型。大幅橫向偏移受既有 15 cm 界限限制，不把不可恢復的案例預設成成功示範。

此批沒有每回合改變摩擦、載重或碎石配置。它增加的是姿態、位置、速度及恢復情境的多樣性，不能稱為已完成物理泛化訓練。

## 資料與切分

正式批次：`logs/rough_bc_demos/diverse_20261116_v2/`。前置測試與中止原型各自保存，沒有偷偷併入正式回合。

- `dataset.json`：預定案例、實際成功／失敗、資料與程式／資產雜湊。
- `progress.json`：目前回合與階段，世界座標僅作觀測進度。
- 每回合 `.npz`／`.json`：原始雷達、因果觀測、教師／執行命令、有效標籤、階段、受擾及獨立世界座標。
- `source_snapshot/`：當次程式與小型設定資產快照；大型 STL 以雜湊追溯。
- `validation.json`、`split_index.json`：收集後核對及訓練索引。

每類六個 train、一個 validation、一個 test；以完整回合與種子分組，從收集前就固定。這不是不同地形的切分。失敗回合照常保留；BC 索引只接納成功、停穩、獨立出口位置通過且有效標籤足夠的完整回合。保留測試的失敗也放進 evaluation_cases，不能只對成功案例計算成功率。

## 159 維觀測與雙動作

| 索引 | 內容 |
|---|---|
| 0:72 | LiDAR 最小距離分箱，除以 10 m |
| 72:144 | 分箱可用遮罩，保留車體角柱遮蔽資訊 |
| 144:148 | 四類路口任務 one-hot |
| 148:150 | 當前量測的局部路口中心，除以 1.5 m |
| 150:152 | 目標出口航向誤差 sin/cos |
| 152:154 | 中心／航向有效旗標 |
| 154:156 | `/odom` 量測線／角速度，除以 0.4／0.6 |
| 156:158 | 上一次實際執行動作，除以 0.4／0.6 |
| 158 | 當前與上一筆掃描的時間間隔，除以 0.2 秒 |

動作為 `[v/0.4, w/0.6]`。沒有把世界位置、世界姿態或 FSM 階段編進策略輸入；FSM 階段只用於診斷、遮罩及後續採樣平衡。`/odom` 是目前 Gazebo 的來源，不代表已校準實車輪速。

`teacher_actions` 是教師回傳的安全命令，`executed_actions` 是最後檢查後執行的命令。可取得的安全檢查前請求另存 `safety_requests_m_s_rad_s`，並以 `safety_request_available` 區分，避免把轉彎准入的旋轉探測當成真正執行動作。

幾何過期、重定位、障礙保護與狀態切換幀不當成有效監督標籤，但仍保留原時間序列。建立歷史視窗時先保留完整回合的時間順序，再使用 label_valid 遮罩；不能先刪除失效幀，也不能跨回合接歷史。

世界位置、z、roll/pitch 僅存於 `*_privileged` 欄位作驗收。沒有接觸感測器，出口位置與停穩檢查不能證明完全無接觸。

## 執行與核對

保留原有 scan／odom／cmd_vel 橋接，另外橋接世界控制、姿態設定與時鐘；不要新增第二個速度發布器：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export IGN_PARTITION=4wd_env_47 GZ_PARTITION=4wd_env_47
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ros2 run ros_gz_bridge parameter_bridge \
  '/world/pig_pen_16units_world/control@ros_gz_interfaces/srv/ControlWorld' \
  '/world/pig_pen_16units_world/set_pose@ros_gz_interfaces/srv/SetEntityPose' \
  '/world/pig_pen_16units_world/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
  --ros-args -r __node:=rough_demo_transport \
  -r /world/pig_pen_16units_world/clock:=/clock
```

在同樣環境變數的另一個終端執行，使用新目錄保留歷史：

```bash
python3 rl_junction/collect_rough_demos.py --output logs/rough_bc_demos/NEW_RUN
python3 rl_junction/validate_rough_demos.py logs/rough_bc_demos/NEW_RUN
```

收集器控制世界播放／暫停；不要另外按播放讓模擬超出指定步數。每回合完成後送零速度、驗證停穩；結束時保留世界暫停。`--case-index 20` 可單獨檢查第一個恢復案例。`--allow-partial` 只適合查進度，不代表資料切分已完整。
