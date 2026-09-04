# E09 Phase-A 实验计划

## 假设
静态 LoRA 将不同 hop/relation-path 查询压入同一低秩更新而产生干扰。若成立，正确结构路由的 4 个 rank-2 basis 在同样 rank-8 参数预算下应优于 static、uniform 和 shuffled control，尤其改善 novel composition。

## 最小矩阵
B1 `static_rank8`；B2 `uniform_basis`；B4 `shuffled_oracle_route`；O1 `oracle_hop_route`；O3 `oracle_relation_path_route`。seeds=43/44，validation-only。

## Gate
最强 O1/O3 相对 B1：overall ≥+2.0pp；novel ≥+2.0pp；seen 下降≤1.0pp；相对 uniform/shuffled ≥+1.0pp；rank-8 参数完全一致。失败即 `STOP_GRAPH_CONDITIONAL_LORA`。
