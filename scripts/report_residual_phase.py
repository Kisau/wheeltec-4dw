#!/usr/bin/env python3
"""Publish the actual completed pilot comparison, never planned outcomes."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();root=a.root
meta=json.loads((a.evaluation/'evaluation.json').read_text());audit=json.loads((root/'audit.json').read_text());bc=json.loads((root/'bc/model.json').read_text())
if not meta['complete'] or not audit['passed']:raise SystemExit('Need complete audited evaluation')
rows=[]
labels=['偏移／右轉進入','偏航／右側返回','低摩擦／左轉進入','加重／左側返回']
for e in meta['episodes']:
 live=e['dynamics']['live_verification'];mode='BC' if e['mode']=='bc' else 'BC＋PPO'
 rows.append(f"| {labels[e['scenario']]} | {mode} | {'成功' if e['success'] else e['reason']} | {e['fsm_correction_frames']} | {e['projection_frames']}/{e['steps']}（{100*e['projection_frames']/e['steps']:.1f}%） | {100*e['exit_offset_m']:.2f} | {live['total_mass_kg']:.2f} / {live['wheel_mu'][0]:.2f} |")
results={m:[e for e in meta['episodes'] if e['mode']==m] for m in ('bc','residual')}
n={m:sum(e['success'] for e in es) for m,es in results.items()}
if n['residual']<=n['bc']:
 conclusion='本輪沒有證據支持用 PPO 取代 BC 作預設。成功率相同或較低時，少量護欄／偏差差異不能證明整體泛化改善。'
else:conclusion='本輪 PPO 成功案例較多，但每條件只有一對，尚未證明一般化；不能據此直接取代 BC 或部署實車。'
fsm_zero=all(e['fsm_correction_frames']==0 for e in meta['episodes'])
fsm_note='本輪所有配對案例的 FSM 修正接管均為 0，因此這批案例無法證明 PPO 能減少原本的 FSM 恢復接管；需要另建確實會觸發合法修正的基準。' if fsm_zero else 'FSM 修正幀數已獨立列出，不能把恢復接管算成模型自行完成。'
s='''# 凍結 BC ＋有界 residual PPO：2026-11-17～18

已完成乾淨 FSM 示範收集、新版 BC 訓練、首輪 512 步 physical Gazebo residual PPO，以及四組獨立配對擾動評估。這是初期流程與效果驗證，非已收斂模型。原有程式、模型與實車服務未替換。

## 示範與 BC

新增四方向起步橫向 ±6 cm、偏航 ±6° 的純 FSM 完整修正／轉彎／出口示範，四回合均完成；486 幀有效標籤，沒有中途瞬移、馬達抑制或定位衝突注入。原資料中的四個人工事件訓練回合已排除；無效定位、障礙與過渡幀不進動作損失。這是起步偏移修正示範，並非所有實車打滑／轉不動的失敗分布。

新版 BC：24 train／4 validation／4 test，局部有效幀 2641／474／649；三種子各 400 epochs，僅 validation 選模，seed 1／epoch 97。新舊 BC 的訓練資料不同，不能把資料清理與 PPO 的影響混算；下表兩組都使用同一個新版 BC。

## 有界 residual PPO

BC eval、requires_grad=False，獨立 PPO 優化器；BC 檔案雜湊在訓練與評估後保持一致。512 步更新確實改動 residual actor，不只是 critic。修正上限線速度 ±0.03 m/s、角速度 ±0.12 rad/s；原地旋轉不加線速度。之後仍經原幾何准入、出口低速、速度與障礙限制。安全等待沒有一般時間罰分，成功須由獨立出口與停穩檢查確認。

訓練原環境快照留在 `ppo/source_snapshot/`。完成後調整日誌以記錄實際執行幀的 owner／phase，並在正式評估前補強同幀觸發障礙時的 residual 遮罩、清除跨回合歷史。評估用同一版本主管比較兩者，原訓練版本與正式評估版本有差異，均保留雜湊，未假稱完全相同。`paired/` 是補強前中止的預檢，不算正式結果。

## 正式配對結果

| 條件／任務 | 模型 | 結果 | FSM 修正幀 | 護欄裁切幀／總幀 | 出口橫向偏差 cm | 實際總質量 kg／輪 μ |
|---|---|---|---:|---|---:|---|
'''+ '\n'.join(rows)+f'''

BC {n['bc']}/4；BC＋PPO {n['residual']}/4。{conclusion}

{fsm_note}

每組相同起點、姿態與物理模型，模式間重啟世界以清除 solver 狀態；輪摩擦四輪 mu1/mu2 與總質量由 Gazebo live entity 的 ModelSdf 元件讀回，非僅引用設定檔。低摩擦 μ=0.65，加重是在原 base 質心均勻分布近似增加 5 kg，慣量同比增加。每種條件只搭配一種任務，無法分離任務與物理條件交互影響，也沒有多種子統計。

護欄裁切仍存在，不能把成功包裝為無介入。裁切次數包含微小的限速／方向修正，不等於接近碰撞次數；本輪尚未比較裁切幅度。沒有接觸感測器，因此出口／停穩通過不等於零碰撞證明。沒有整條六支線巡邏驗證，也未部署 Xavier。

## 稽核與下一步

`audit.json` 核對乾淨標籤、檔案、BC 凍結、actor 更新、residual 上限、主管接管、配對完整性與實際物理參數。正式評估資料位於 `{a.evaluation.name}/`，逐幀日誌與每次 live SDF、來源快照均保留。

下一輪先固定這組作開發基準，另建未見的四任務×各物理條件×多種子評估；訓練預算至少跨越更多完整回合，再依 validation 決定 checkpoint。仍以成功率、出口偏差、FSM 接管與護欄比例共同判斷，不單純獎勵更快或取消停車。此輪條件已用於觀察結果，後續調參不能再稱它們為未見測試。

動態 remove/create GPU 雷達車模曾觸發 Ogre2 斷言，該嘗試不列物理測試通過；正式流程改為啟動時嵌入車模並管理本實驗 server。結束已恢復名義物理場景、暫停且沒有活動控制器。

操作與架構：[RESIDUAL_PHASE.md](../../../rl_junction/RESIDUAL_PHASE.md)。
'''
(root/'RESULTS.md').write_text(s);print(root/'RESULTS.md')
