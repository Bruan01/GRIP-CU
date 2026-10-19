# Dataset roles

更新日期：2026-10-19

当前主线只保留 NELL23K 和 FB15K237-2，两个数据集都跑同一组三臂比较：Original GRIP/B1、Random-9 + InfoNCE、ErrorMemory-9 + InfoNCE。

| 数据集 | 本地目录 | 主要作用 | 关系标签 |
|---|---|---|---|
| NELL23K | `04_benchmark/data/nell23k/` | 原始 GRIP 复现锚点、长尾实体和多关系候选学习 | `concept:*`，直接读取 `train.txt` |
| FB15K237-2 | `04_benchmark/data/fb15k237_2/` | 外部多关系 KG 验证、关系文本和候选边界泛化 | `rel2text.json` 映射后的文本关系 |

## 数据目录合同

每个数据集必须包含：

- `raw/train.txt`、`raw/valid.txt`、`raw/test.txt`；
- `processed/processed_val.json` 和 `processed/processed_test.json`；
- `tasks/` 下的完整 Original GRIP task JSON，不能用 smoke 或截断文件替代正式训练。

ErrorMemory-9 的错误记忆、候选分数和已知事实过滤只能使用 `raw/train.txt`。`valid.txt` 和 `test.txt` 只能进入评估。

## FB15K237-2 说明

这里使用 GRIP 代码中的 `fb15k237_2` 版本，而不是另一个空的下载占位目录 `fb15k237/`。FB 的 raw relation ID 通过 `raw/rel2text.json` 转成 task 中使用的文本关系；runner 需要显式传入这个映射文件。
