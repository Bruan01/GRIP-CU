# Dataset Roles

更新日期：2026-08-28

## 总体原则

六个数据集不承担相同作用。CLEGR 和 Scene Graph 提供主要机制证据，四个 KG 验证真实闭卷调用和关系组合。

| 数据集 | 核心角色 | 主要实验 |
|---|---|---|
| CLEGR | 主机制数据集 | hop 外推、图规模 OOD、frontier、因果干预 |
| Scene Graph | 自然语义图验证 | 空间/对象关系组合、场景间泛化 |
| FB15K237 | 多关系 KG | relation composition、路径 OOD、多答案 |
| WN18RR | 层次语义 KG | 上下位、逆关系、层次传播 |
| CoDEx-Medium | 稀疏复杂 KG | 低频关系、度数偏移、复杂路径 |
| NELL23K | 当前复现锚点 | 闭卷事实调用、长尾实体、多跳扩展 |

## CLEGR

### 用途

- 控制真实最短路径长度；
- 获得每一步 BFS frontier ground truth；
- 控制节点数、平均度、分支因子、环比例；
- 生成训练长度之外的测试实例。

### 推荐切分

- ID train：1–2 hop，小图；
- ID validation：1–2 hop，小图；
- Length OOD：3–8 hop；
- Size OOD：更大节点数；
- Density OOD：不同平均度；
- Label OOD：全部节点随机重命名。

## Scene Graph

### 用途

测试符号机制能否迁移到自然场景关系，包括：

- left/right；
- above/below；
- inside/contains；
- overlap/near；
- object–attribute–relation composition。

### 注意

必须避免问题文本直接泄漏目标对象，并区分单跳属性问答与真正关系组合。

## FB15K237 / WN18RR / CoDEx-Medium / NELL23K

对每个 KG 额外构造 path-query 集合：

1. 从源实体采样关系约束路径；
2. 计算真实最短有效路径；
3. 删除存在一跳答案捷径的样本；
4. 保存每一步 frontier；
5. 创建 relation-sequence OOD split；
6. 创建 entity alias / random label 版本；
7. 对多答案问题保存完整目标集合。

## 扩展数据集

WebQSP 和 CWQ 可作为后续 KGQA 外部验证，但不进入第一轮主实验。只有六数据集主协议稳定后才加入，避免范围失控。
