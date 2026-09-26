# YCB-V / BOP-YCB-V 数据审计（第一版）

## 结论

当前 `data/bop/ycbv` 是 **BOP 转换格式**，不是原始连续 YCB-Video。已下载的是 BOP19 test target 子集：12 个场景、900 张 RGB 关键帧、4,125 个 GT 实例、21 个对象。每个场景恰有 75 张图，但相邻可用 `frame_id` 的间隔为 1–912，因此该子集适合标准单帧位姿协议，不足以支撑本文的连续 tracking/relocalization 主结论。

审计命令：

```bash
python scripts/analyze_dataset.py \
  --dataset-root data/bop/ycbv \
  --split test \
  --output analysis/ycbv_bop19_audit_v1 \
  --seed 2027 \
  --visual-samples 24
```

机器可读结果位于 `analysis/ycbv_bop19_audit_v1/`。脚本会拒绝覆盖非空输出目录，以保留实验可追溯性。

## 标注语义与字段边界

- `visibility_fraction` 直接使用 BOP `visib_fract = px_count_visib / px_count_all`，`invisible_fraction = 1 - visibility_fraction`。
- `px_count_valid` 表示目标轮廓中具有有效深度测量的像素，不等于“落在图像内的完整投影面积”。所以当前数据不能可靠拆分 occlusion 与 truncation；两字段均为 `NaN`。
- `translation_delta` 和 `rotation_delta_deg` 是同一场景、同一对象在相邻**可用关键帧**之间的 GT 差值。它们不是逐视频帧速度；无 timestamp/FPS，速度字段为 `NaN`。
- `sharpness_score` 是目标 crop 的灰度 Laplacian 方差，只是 sharpness proxy，不能解释为物理 motion blur 强度。
- 当前没有 posed reference manifest，reference viewpoint gap 全部为 `NaN`。脚本已支持 `--reference-manifest`，接入真实 reference bank 后才计算。
- clutter proxy 来自同帧可见 bbox 的 IoU；没有把它描述为真实遮挡比例。

字段来源逐行保存在 `attribute_source`，未知量没有被估造。

## 自然分布

以下均是 test subset 的描述统计，**不能作为最终 bucket 阈值**：

| 属性 | 最小值 | Q1 | 中位数 | Q3 | 最大值 |
|---|---:|---:|---:|---:|---:|
| visibility fraction | 0.075 | 0.804 | 0.990 | 0.999 | 1.000 |
| bbox area ratio | 0.0089 | 0.0469 | 0.0734 | 0.1144 | 0.3457 |
| visible area ratio | 0.0024 | 0.0298 | 0.0459 | 0.0724 | 0.2265 |
| sampled translation delta (m) | 0.00017 | 0.00318 | 0.00694 | 0.01307 | 0.1730 |
| sampled rotation delta (deg) | 0.000 | 0.281 | 0.588 | 1.155 | 34.551 |
| sharpness proxy | 244.8 | 1847.1 | 2919.6 | 4274.8 | 10911.6 |
| max bbox IoU | 0.000 | 0.0658 | 0.1436 | 0.2497 | 0.6221 |

用于理解尾部规模的描述性计数：visibility ≤ 0.50 有 118/4,125（2.86%），≤ 0.25 有 23/4,125（0.56%）；bbox area ratio ≤ 0.02 有 55/4,125（1.33%）。这些数值只描述覆盖情况，不定义 “heavy occlusion” 或 “small object”。

固定种子抽取的 24 张可视化已人工检查：目标框总体与图像中的实例吻合；较低 visibility 的样本视觉上呈现遮挡、边界截断或两者混合，符合“不强行拆分”的处理。检查用 contact sheet 为 `analysis/ycbv_bop19_audit_v1/figures/visual_audit_contact_sheet.jpg`。

## 对十个审计问题的回答

1. **Visibility**：分布高度偏向完全可见，中位数 0.990，但存在 118 个 ≤0.50 的尾部样本。
2. **Severe occlusion 是否充足**：无法从当前 BOP 字段把 severe occlusion 与 truncation 分离；低 visibility 尾部总量有限，只能研究“低可见性”。
3. **Truncation 是否充足**：不可判定，需要原始数据的完整投影/边界标注或可验证的渲染流程。
4. **Rotation motion**：可用关键帧差值范围 0–34.55°，但时间间隔不等，不能作为真实角速度或连续跟踪难度。
5. **Translation motion**：可用关键帧差值范围 0.00017–0.173 m，同样不能解释为线速度。
6. **Small object 是否充足**：存在尺度变化，但极小实例稀少（area ratio ≤0.02 仅 55 个）。尺度 quantile 分析可做，极小目标结论需更多数据。
7. **Reference viewpoint gap**：当前 reference bank 缺失，不能回答。
8. **Baseline failures 区域**：见 `docs/failure_analysis.md`；目前只覆盖 object 5 的两个场景，不能推广到全数据集。
9. **可支撑自然主实验的因素**：visibility、object scale、bbox overlap/sharpness proxy 有全量字段；但须在独立 validation split 冻结 quantile。
10. **可能需要补充数据/受控实验的因素**：纯 occlusion、truncation、真实速度、reference gap、极小目标，以及连续 long-term tracking。优先补原始连续序列与 posed references；synthetic stress test 只作补充。

## 下一步的数据前提

在创建 `configs/dataset_stratification.yaml` 前，需要独立的 validation sequences。最终 Q1/Q2/Q3 必须仅由 validation attributes 生成并冻结，test 只按冻结规则报告。连续跟踪协议还需要原始 YCB-V 视频帧（或其他连续 RGB 数据）、明确帧序/FPS，以及合法的 posed reference bank。

