# 平坦 ㄇ 字形走道

`u_corridor_flat.world` 是獨立的 Gazebo 測試世界，僅含平坦地板、實體碰撞牆與光源。
無碎石、斜坡、角柱、動態障礙或初始偏移。

尺寸以走道中心線計：左右側各 23 m、頂端 22 m，淨寬 2.4 m。
牆厚 0.1 m、高 1 m。外側淨空包絡為 24.4 × 25.4 m。
兩底端封閉；端點中心距端牆 1.2 m，實際車身餘量須再扣半車長。

```text
 B(0,11.7) ──── 22 m ──── C(22,11.7)
     │                        │
    23 m                     23 m
     │                        │
 A(0,-11.3)               D(22,-11.3)
```

路線設定：`maps/u_corridor/route.json`。單程 68 m、往返 136 m。
去程前進 A→B→C→D，回程倒退 D→C→B→A。
此 JSON 僅描述路線，不會自動啟動 FSM。

## 專用純 FSM

獨立控制器：`rl_pig_pen/u_corridor_fsm.py`；ROS 執行器：
`scripts/run_u_corridor_fsm.py`。不載入舊巡邏 FSM、BC 或 PPO。
以 A 點初始車頭方向建立局部里程計座標：A=(0,0)、B=(23,0)、
C=(23,-22)、D=(0,-22)。需要從 A 點重新產生的車輛啟動。
此版使用已知地圖的雷達牆面配對定位，里程計提供兩次掃描間的運動預測，
雷達另外提供避障檢查。純幾何最小平方計算，不需要模型訓練，亦非無地圖 SLAM。
只配對牆的道路側表面，避免把薄牆背面當成道路邊界。
配對點不足、矩陣秩不足、殘差過大或單次修正跳躍過大時立即停止。
不得把 Gazebo 世界座標餵回控制器；世界座標僅供獨立驗證。

每段依序執行 `ALIGN`（停穩後原地對正）→`DRIVE`（循線、接近目標減速）
→`SETTLE`（確認量測速度近零並停留至少 1 秒）。D 點切換負線速度，
沿 D→C→B→A 倒退；A 點進入 `DONE`。B、C 仍需轉動車身，沒有 180° 掉頭。
線速度上限預設 0.4 m/s、角速度上限 0.5 rad/s。

雷達與里程計接收逾時、時間戳不一致、障礙侵入行進包絡、旋轉空間不足、
偏離路線超過 0.55 m、持續 3 模擬秒無進展等情況均進入鎖定 `FAILED`，
持續零速度至退出。障礙清除不會自動恢復；須確認原因並重新從 A 啟動。
物理停止判斷使用里程計量測速度，不能視為硬體停車保證。

另開終端執行（唯一速度發布器）：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=48 IGN_PARTITION=4wd_env_48 GZ_PARTITION=4wd_env_48
python3 scripts/run_u_corridor_fsm.py --speed .4 --output logs/u_corridor/my_new_run
```

`--output` 必須是新目錄；保留逐步 JSONL、狀態 topic `/u_corridor/status`、
終端摘要 `result.json`。Ctrl-C 送零速度後退出。不要在移動中重啟控制器，
本版不支援斷點續走。其他終端的 ROS_LOCALHOST_ONLY、RMW 應與啟動器一致。

唯讀獨立驗證（在控制器啟動前執行，使用相同環境）：

```bash
python3 scripts/monitor_u_corridor.py --output logs/u_corridor/my_new_audit
```

驗證器從 Gazebo 世界座標檢查六次到點誤差 <0.15 m，並記錄距牆及傾角。
控制器不讀世界座標；驗證器不發布速度。距牆檢查不能替代接觸感測器。

控制與失效測試：

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s rl_pig_pen/tests -p 'test_u_corridor*.py' -v
```

地圖單獨開啟（不產生車輛）：

```bash
IGN_PARTITION=4wd_env_48 GZ_PARTITION=4wd_env_48 ign gazebo worlds/u_corridor_flat.world
```

使用現有啟動器產生車輛及 ROS 橋接（不啟動巡邏控制器）：

```bash
bash scripts/launch_clean.sh --env-id 48 --lidar-only --rtf 1 worlds/u_corridor_flat.world
```

初始位置沿用啟動器的 (0,-11.3,0.05)，車頭朝 +Y，位於 A 點中心線。
其他控制與觀測終端須使用 ROS_DOMAIN_ID=48 及相同 Gazebo partition。
請勿同時在此 domain 啟動多個速度發布器。

重新產生及檢查：

```bash
python3 scripts/build_u_corridor.py
ign sdf -k worlds/u_corridor_flat.world
```
