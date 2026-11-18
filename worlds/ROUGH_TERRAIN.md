# 實體細碎石碰撞與架高豬舍場景

使用 `pig_pen_16units_rough.world`；由含雙 ㄇ 形角柱的基準地圖產生。
舊平地地圖保留供比較。

## 實際地形

- 主幹道與 16 個豬舍抬高 0.06 m，角柱與豬模型同步抬高。
- 六個入口斜坡：高 0.06 m、水平長 0.15 m、寬 0.99 m，約 21.8 度。
  高端 X=±0.60，低端 X=±0.75；支線維持低處。
- 碎石直徑 5–10 mm，露出高度 1–3 mm。每顆是具有封閉表面的立體
  六角雙錐，底部嵌入地面，頂部由物理引擎參與輪胎接觸。
- 全圖約 272,103 顆，合計 3,301,284 個碰撞三角形。顆粒合併進各道路
  的 `_gravel.stl`，避免建立數十萬個獨立剛體／ROS 實體。
- 道路 visual 與 collision 皆指向同一個實體顆粒 STL。碎石高低由實際
  三維三角形表達，貼圖、法線貼圖與原本 8 cm 採樣網格不負責碎石碰撞。
- 底面已回到平面；粗糙接觸來自立體顆粒，不再來自大尺度平滑起伏。
- 固定嵌入式碎石：會碰撞，但不滾動、不位移、不模擬鬆散碎石的沉陷。
  依目前「露出 1–3 mm」條件建模，不能聲稱已還原鬆散顆粒土的力學。
- 分區摩擦係數暫用 0.55–0.85，尚未依實車量測校準。
- 物理網格增加，實際模擬倍率受 CPU／接觸計算與渲染成本影響。

## 生成及啟動

```bash
python3 scripts/build_rough_pen_world.py

env ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  bash scripts/launch_clean.sh --env-id 47 --lidar-only --rtf 2 \
  --spawn-z 0.11 worlds/pig_pen_16units_rough.world
```

預設每個 128 mm 方形區域選 96 顆顆粒。`--grains-per-tile` 可改密度；
0 是無顆粒的平面對照，不應拿低密度測試結果宣稱預設全圖已驗證。
完整尺寸與顆粒數在 `pig_pen_16units_rough.terrain.json`。

## 已完成驗證（2026-11-12）

1. 六項幾何／生成測試：顆粒封閉性、實際 STL 中的直徑及突出高度、
   視覺與碰撞使用相同 STL、斜坡銜接、架高及資源路徑。
   單元測試採較小顆粒密度縮短生成時間，完整場景另有實際載入驗證。
2. 獨立 DART 物理接觸對照：半徑 2 mm 小球以垂直移動約束分別落在
   碎石尖端與平面。碎石預期凸出 1.66508 mm，測得球心高度差
   1.66535 mm，誤差約 0.00027 mm。這驗證引擎讀到真實顆粒碰撞面，
   不是僅靠材質或修改車身姿態。
3. 完整 27 萬顆場景已載入，雷達正常輸出 1,640 射線。輪胎低速短距離
   測試前進約 23 cm，實測車身 Z 約 0.096477–0.096503 m，俯仰約
   -0.0794–0.0559 度、側傾約 -0.0631–-0.0244 度；結束送零速度。
   這不是完整爬坡或巡邏驗證，也未以此推論碎石一定造成嚴重打滑。

```bash
python3 -m unittest discover -s scripts/tests -p test_rough_pen_world.py -v
python3 scripts/tests/gravel_contact_fixture.py --output /tmp/gravel_contact_probe
```

接觸對照 fixture 產生 `probe.world` 與 `expected.json`。可在獨立 partition
以 `ign gazebo -s -r /tmp/gravel_contact_probe/probe.world` 啟動，讀取
`/world/gravel_contact_probe/dynamic_pose/info` 中 `grain_probe`、`flat_probe`
的世界 Z 座標。兩者差值應符合 expected.json 的 grain_height_m。

此次實測紀錄：`logs/env_47/physical_gravel_validation/contact_probe.json`、
`wheel_motion_summary.json`。完整巡邏、上下坡以及現場摩擦／打滑校準仍待驗證。
