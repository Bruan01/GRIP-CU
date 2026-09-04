# 服务器实验需求

1. 固定 commit，Original GRIP 只读，服务器不开发代码。
2. 仅打开 train/validation；严禁 test 文件、test prediction 和 test 选模。
3. 首轮只跑 B1/B2/B4/O1/O3 × seeds 43/44，共 10 runs。
4. rank-8 方法 trainable parameters 相同；审计 optimizer steps、answer tokens、max length、decoder。
5. O1/O3/B4 的 gold route 仅是 oracle diagnostic，runtime audit 必须标记，不能描述为部署方法。
6. gate 失败立即停止；不自动开发 learned/Bayesian router，不上 7B。
7. 回传 environment、config SHA256、data/route audit、parameter/training accounting、validation predictions、metrics、gradient probe、adapter SHA256、suite 与 manifest。
