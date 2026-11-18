# 凍結 BC ＋有界 residual PPO

這是獨立實驗，不取代舊模型、控制入口或實車服務。PPO 學習方法與 Transformer 架構分開；本版使用小型 MLP residual，保留 phase BC 與原路口安全主管。

## 資料

`collect_clean_fsm.py` 收集四方向、起步橫向偏移 ±6 cm／偏航 ±6° 的完整 FSM 修正、轉彎與出彎；沒有中途瞬移、抑制馬達或人工定位衝突。僅成功、獨立出口／停穩確認的回合可用，且無效定位、障礙、過渡與失敗標籤不進損失。

`prepare_clean_bc_sources.py` 從原 diverse 資料排除含人工事件的訓練回合，保留原 validation／test 的完整回合切分。它們仍含原有擾動，作評估而非訓練示範；新的四回合均屬 train，不將已測的成功／失敗紀錄事後塞回 test。訓練仍以 train 正規化、validation 選 checkpoint，舊權重不覆寫。

## PPO

`residual_phase_env.py` 使用 Gazebo 物理碎石地圖，8×168 BC 歷史加一個當前模型可接管旗標。世界座標僅作回合出口與傾倒評分，不進 actor。BC eval 模式、requires_grad=False、獨立權重雜湊；PPO 優化器只包含 residual 策略／價值網路。

- 動作：`BC + clip(residual, -1, 1) × [0.03 m/s, 0.12 rad/s]`。
- 原地旋轉 residual 線速度為零；之後再經原速度、旋轉准入、出口低速、障礙與資料檢查。
- 模型不可接管時，PPO 動作忽略，由原 FSM／停止流程處理。
- 依當前階段的幾何誤差改善給獎勵，不以速度大小獎勵。獨立出口成功才有終點獎勵；無效定位／安全等待不受一般時間罰分，保留停車能力。
- 不把 teacher action 當作 PPO actor 輸入，也沒有取消停車或限速換取成功率。

首輪訓練只驗證流程與初期效果，不代表收斂。模型另存 `residual_ppo.zip`，與對應 BC 的 SHA256 綁定。使用 `train_residual_phase.py --bc <model.pt> --output <新目錄> --steps 512`。依 PHASE_JUNCTION_BC.md 設定 domain 47／partition 4wd_env_47 與 ROS 橋接；避免任何其他速度發布器與 GUI play。

## 物理條件與評估

`residual_scene.py` 每次啟動時把已轉換的車模嵌入世界；`residual_dynamics.py` 修改四輪 mu1／mu2，或在原車體質心以均勻分布近似增加 base 質量並同比縮放慣量。這不是不明位置的真實貨物慣性模型。

本機動態刪除 GPU 雷達車模曾觸發 Ogre2 斷言，不能沿用 remove/create 作每回合隨機化；改成管理本實驗伺服器的重啟流程。只停止 server_state.json 指向且命令路徑符合本實驗的 PID，不做全機 pkill。

配對比較 BC 與 BC＋PPO：新起步偏移、新角度、低摩擦 μ=0.65、額外載重 5 kg。記錄成功、出口偏差、FSM 修正、護欄裁切、障礙與停止。每條件目前只一種任務／一對，不能分離任務與條件的交互影響，更不能推論統計顯著性；大規模評估需四任務×各條件×多種子。

結果見 [2026-11-17～18 實測報告](../logs/residual_phase/20261117/RESULTS.md)，已執行與未執行項目依實際紀錄判定。固定物理碎石地形、沒有接觸感測器；安全停車不等於已確認實際零碰撞，尚未部署實車。

正式評估使用 `evaluate_residual_phase.py --bc <model.pt> --ppo <residual_ppo.zip> --scene-root <實驗根目錄> --output <新目錄>`；中斷可加 `--resume`，只允許相同模型、來源與固定計畫。`paired_v2/` 為補強障礙遮罩後的正式評估，`paired/` 是中止的預檢。
