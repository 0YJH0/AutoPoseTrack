# Base Tracker 自然失败分析（第一版）

## 分析范围

现有可复现实验是 `outputs/ycbv_object5_rollout_150/rollout.csv`：object 5、场景 000050/000052、共 150 个 BOP19 稀疏关键帧。150 条推理均返回 `status=ok`，因此当前没有可验证的二值 GT failure label。本报告不人为设置“失败阈值”，而是将连续 `pose_error_m` 与自然属性对齐。

复现命令：

```bash
python scripts/analyze_tracking_failures.py \
  --predictions outputs/ycbv_object5_rollout_150/rollout.csv \
  --attributes analysis/ycbv_bop19_audit_v1/per_frame_attributes.csv \
  --output analysis/ycbv_bop19_audit_v1/baseline_object5 \
  --error-field pose_error_m
```

结果会通过 `(sequence_id, frame_id, object_id)` 严格 join；若一个对象在同帧存在多个实例而缺少 `gt_id`，脚本会报歧义而不是静默匹配。

## 误差与属性对齐结果

位姿误差均值 0.0191 m，中位数 0.0185 m，Q75 0.0230 m，Q90 0.0260 m，最大值 0.0972 m。最大误差发生在 scene 000052/frame 388，明显高于其余样本。

连续误差与属性的 Spearman 相关系数如下：

| 自然属性 | 样本数 | ρ(error, attribute) |
|---|---:|---:|
| bbox area ratio | 150 | -0.494 |
| visible area ratio | 150 | -0.451 |
| max bbox overlap | 150 | +0.473 |
| sharpness proxy | 150 | +0.461 |
| visibility fraction | 150 | -0.287 |
| sampled translation delta | 148 | +0.118 |
| sampled rotation delta | 148 | -0.029 |
| reference viewpoint gap | 0 | unavailable |

这批样本中，较小目标和较高 bbox overlap 与更高误差有中等单调关联。visibility 的关联较弱，原因之一是 object 5 的 visibility 范围非常窄（0.965–1.000），不包含全数据集的低可见尾部。sharpness proxy 呈正相关，说明它在该对象上可能主要反映纹理/边缘量，而非 blur；不能据此声称“越清晰越差”。运动相关结果也不能当作速度结论，因为关键帧间隔不统一。

## 当前不能声称的结论

- 这不是连续视频 rollout；局部更新跨越 1–数百个原始 frame id，不能用于 long-term tracking survival、recovery latency 或 lost-time ratio。
- 没有二值 failure GT，不能报告 AUROC/AUPRC/F1，也不能验证 recoverability predictor。
- 只有一个对象、两个场景，且高度可见；不能断言 visibility、occlusion 或 reference gap 是主要失败原因。
- 相关性是发现假设，不是因果证据；最大误差点也可能主导均值。

## 数据驱动的下一阶段方案

1. **先补连续协议数据**：获取原始连续 RGB 帧、FPS/时间戳和 sequence-level train/validation/test split；保留当前 BOP19 子集用于 Protocol A 单帧位姿评价。
2. **建立 reference bank**：记录每张 reference 的已知 SE(3)，重新审计 nearest/mean viewpoint gap；所有方法共享相同 references。
3. **在 validation 上冻结分层**：优先对 visibility、scale、overlap、sharpness、真实角/线速度和 reference gap 使用 quartile；冻结后再运行 test，绝不依据 test improvement 调阈值。
4. **Base/Heuristic/Ours 同协议比较**：先输出连续误差曲线与全 quantile trend，再加入有明确 pose-correctness 协议的 failure detection 和 multi-step recoverability label。
5. **受控实验后置**：recoverable basin 用于机制分析；只有自然数据覆盖不足时，才补 temporary occlusion、blur、mask dropout 等 stress test，并保留原始属性和随机种子。

