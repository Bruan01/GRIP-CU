# 基线修复与方向决策计划（修正版）

更新日期：2026-09-03

> ✅ **Phase 1 决策门已通过（2026-09-03 quick01）**：0.5B 上按论文配方（MLP LoRA 全层 + 两阶段训练），
> **correct adapter 47.8% > base model 22.5%**（test 48.0% vs 21.7%）。「图写进参数」在本环境成立。
> 详见 `08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/results/runs/quick01_storage_quick/RESULT.md`。
> 下一步：下载 7B，做完整复现。

替代之前对 Hard-Negative / RecurrentGRIP 的乐观判断。本计划回答一个核心问题：

> 在「图能不能写进参数」这一步本身都没被证明成立的当下，下一步到底该做什么。

---

## 0. 现状与根因（修正版）

### 0.1 两条支线的真实状态

| 方向 | 状态 | 判定依据 |
|---|---|---|
| Hard-Negative Contrastive | **放弃** | H2 在 base model 与 adapter 两层都被证伪（结构负样本不更难），与训练好坏无关 |
| RecurrentGRIP | **未检验，非已失败** | 初步机制指标（best-K↔hop Spearman=0.0）为阴性，但被「没训好 + benchmark 用错」双重混杂，核心假设未被证明 |

### 0.2 根因：pilot 配置严重偏离论文配方

`RecurrentGRIP/v1_1` 的 pilot 与论文 GRIP 几乎每一项都错位：

| 配置项 | 论文 GRIP（NELL23K，EM=87.74%） | v1.1 pilot（correct adapter 4.5%） |
|---|---|---|
| 基础模型 | Qwen2.5-**7B**-Instruct | Qwen2.5-**0.5B**-Instruct |
| LoRA 挂载模块 | **MLP**（down_proj/up_proj/gate_proj） | **注意力**（q/k/v_proj） |
| LoRA 层范围 | 全层 | 仅第 12 层（`layers_to_transform=[12]`）|
| LoRA rank/alpha | 4 / 8 | 4 / 32 |
| 训练任务量 | 8000 context + 2000 reasoning + 6000 summary | 512 QA |
| Stage 2 轮数 | 1–10 epoch | 1 epoch |
| 有效 batch | 512 | 8（batch1 × grad_accum8）|
| 学习率 | 1e-3 | 2e-4 |
| Stage 1 早停 loss | 0.15 | 未收敛（loss 停在 5~6）|

**两个致命点**：
1. **图知识写错了地方**——论文存在 MLP（前馈层），pilot 存在注意力层。Transformer 的事实知识主要分布在 MLP，注意力 LoRA 很难存图。
2. **根本没训进去**——0.5B + 1 epoch + 512 样本 + 有效 batch 8，loss 停在 5~6（困惑度 ≈ 400），等于 Stage 1 记忆阶段没发生。

**结论**：`4.5% < 22.7%` 是配置错误，不是 GRIP 方法失效，更不证明 GRIP 论文为假。当前所有增量方向的共同卡点，是「图写进参数」这一步在本环境从未被成功复现。

---

## 1. 唯一硬门槛

在谈任何增量（RecurrentGRIP 机制、新 loss、新负样本）之前，必须先跨过这一关：

> **用论文配方复现出「adapter 命中率 > base model」的 GRIP 基线。**

判定标准（NELL23K，正确 adapter vs none）：
- ✅ 通过：correct adapter 的 EM 明显高于 base model（例如 > 30%，接近论文 87.74% 的缩水版）。
- ❌ 未通过：correct adapter ≤ none（像现在这样）→ 图没写进去，任何机制/增量实验都是无源之水，立即止损。

---

## 2. Phase 0：复现原始 GRIP 基线（不做创新，只复现）

目标：在现有 24GB RTX 3090 上，按论文配方把 GRIP 训出来。

### 2.1 必须对齐的论文配方（按优先级）

1. **模型换 7B**：`Qwen2.5-7B-Instruct`（硬件支持，论文同 GPU 类跑过）。
2. **LoRA 换 MLP 模块**：`target_modules = [down_proj, up_proj, gate_proj]`，全层挂载（不 `layers_to_transform` 限制单层）。
3. **两阶段训练**：
   - Stage 1：context 记忆，训练到 loss ≤ 0.15（早停），而不是固定 1 epoch；
   - Stage 2：QA，1–10 epoch。
4. **有效 batch 512**、**LR 1e-3**、weight decay 1e-4、max grad norm 1.0。
5. 训练任务量对齐（context/reasoning/summary 的量按 3090 能承受的规模等比例缩小，但相对比例保持 8000:2000:6000）。

### 2.2 验证脚本

- 用 `13_base_method/grip-exp`（原始 GRIP，保持 clean），按上述配方跑 NELL23K。
- 输出：correct adapter vs none 的 EM，以及 Stage 1 是否真的把 loss 降到 0.15 附近。

### 2.3 成功标准

- Stage 1 loss 收敛到 ≤ 0.15（图真的写进去了）；
- correct adapter EM > none EM。

---

## 3. Phase 1：决策门

- **通过** → 「图写进参数」成立，进入 Phase 2。
- **未通过** → 两个可能，需分开排查：
  1. **训练还没到位**：继续加 epoch / 加数据 / 调 batch，直到 Stage 1 收敛；
  2. **LoRA r=4 容量不够存 24k 边的图**：尝试 r=8/16，或分层存储。
- 若反复排查仍无法让 adapter > none，则诚实结论是「图写进参数在当前设置下不可靠」，**整个 GRIP 增量方向（含 RecurrentGRIP）都需要重新评估**，而不是硬发论文。

---

## 4. Phase 2：RecurrentGRIP 机制检验（前提满足后）

只有在 Phase 1 通过后才做。核心教训：**先证明 storage 成立，再谈 execution。**

### 4.1 关键修正：区分 storage 设计与 execution 设计

当前 v1.1 把 LoRA 改成「单层注意力」是为了做 recurrence，但这一步**破坏了 storage**（图存不进去）。正确顺序是：

1. 先让图 LoRA 能存图（沿用论文的 MLP 全层方案）；
2. 再把「递归执行」作为**额外的执行机制**叠加，而不是用递归去替换存储结构。

### 4.2 机制实验（按序）

1. **CLEGR** 上测 H1（最优 recurrence K ↔ 真实 hop 正相关）——NELL23K 关系预测不是多跳任务，机制结论必须靠 CLEGR。
2. H2（frontier probe）：冻结模型，线性 probe 第 k 轮隐状态是否能预测第 k 跳 frontier。
3. H4（activation patching）：正确/错误轨迹双向 patch 是否改变答案。
4. H5（storage–execution 分离）：adapter swap 改变答案与 frontier、executor swap 改变执行质量。

### 4.3 失败即止损

若 storage 已成立、但 H1 在 CLEGR 上仍无对齐信号（Spearman≈0），则「图可被逐跳执行」这一核心假设被证伪，RecurrentGRIP 方向终止，转而把「storage 成立但 execution 不成立」本身作为诚实结论。

---

## 5. Phase 3：论文（通过 Phase 2 后）

沿用已有 `11_paper/paper_outline.md`：主贡献不是「超过 GRIP 分数」，而是「内化图的可执行性边界 + 机制证据 + 与 KG-RAG 成本边界」。

---

## 6. 风险与止损条件汇总

| 检查点 | 通过条件 | 未通过则 |
|---|---|---|
| Phase 0 Stage 1 | context loss ≤ 0.15 | 排查训练配方，不进入 Phase 2 |
| Phase 1 | correct adapter EM > none | 停止 GRIP 增量方向，重估 |
| Phase 2 H1（CLEGR）| best-K ↔ hop 显著正相关 | RecurrentGRIP 终止，转「storage≠execution」结论 |

---

## 7. 一句话结论

**当前唯一正确的一步，不是再想新增量点，而是先用论文配方把 GRIP 基线复现出来——这是所有后续工作的地基。地基之前的一切机制/增量讨论都是悬空的。**

### 立即执行清单

- [x] 读 `13_base_method/grip-exp` 的 `shells/qwen_grip_inf/nell23k_single3090.sh` 与 `02_train_nell23k_lora.sh`，确认论文完整训练命令。
- [x] 便宜的 0.5B 快速验证（MLP LoRA 全层 + 两阶段训练）——**PASS：correct 47.8% > none 22.5%**。
- [ ] 在原始 GRIP（不碰 RecurrentGRIP）上用 7B + MLP LoRA + 两阶段训练复现 NELL23K（下一步，需下载 ~15GB 7B 模型）。
- [ ] 记录 Stage 1 loss 是否收敛、correct vs none EM。
- [x] 依据 Phase 1 决策门决定是否继续 RecurrentGRIP——**通过，继续**。
