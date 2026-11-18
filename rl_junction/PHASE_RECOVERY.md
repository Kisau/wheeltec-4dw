# 階段 BC 的停止後牆面恢復

此版只新增 `rl_pig_pen/phase_junction_recovery.py` 子類別及獨立模擬入口 `control_modes_right_then_left_phase_recovery.py`，沿用已訓練的 phase BC 權重；不修改既有模型合約、原控制器或實車服務。

原失敗案例的完整四開口候選突然橫移約 0.9 m，連續三幀與原路口衝突後停止；後續完整路口偵測無法成立。停止有必要，問題是原路口的當前牆面仍可被追蹤，舊恢復流程未使用這條證據。

新增恢復只適用 CENTER／RETURN_CENTER 的 `junction_geometry_conflict`：

1. 保留原停止，要求新鮮里程計與量測靜止。
2. 用已取得路口的牆寬／軸向，對當前掃描追蹤至少三面、跨兩軸的牆；沿用原匹配、歧義、殘差、位移、時間間隔檢查。
3. 連續三幀一致後，仍先回傳零速度，再恢復原階段。只保留已確認的牆面身份，不積分命令或里程計推算中心。
4. 同一階段內優先追蹤已確認牆面；若不可追蹤，回到原偵測／停止流程。沒有放寬信心、中心允許區或障礙條件。

資料重播與 Gazebo 閉迴路分開記錄。原失敗掃描重播可驗證第 21 幀停止、第 22～24 幀確認並恢復；固定軌跡重播不能證明出彎成功。Gazebo 相同初始條件重跑也可能因物理取樣差異未重現衝突，不能把沒有觸發恢復的成功當成恢復機制成效。因此另以明確標記的衝突停車故障注入驗證完整恢復路徑。

紀錄與實際結果見 [本次報告](../logs/phase_junction_recovery/20261117/RESULTS.md)。

重現：沿用 PHASE_JUNCTION_BC.md 的隔離 Gazebo 環境與橋接，改用 `evaluate_phase_recovery.py`；`--stage left-regression` 測三個左轉進場，`--stage recovery-safety` 測一次衝突停車注入及兩次持續動作抑制。每次 `--output` 必須使用新目錄。完成後執行 `python3 scripts/audit_phase_recovery.py <輸出目錄>`。
