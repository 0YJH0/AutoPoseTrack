# Reference-guided initialization pilot

Configuration: official SAM ViT-B automatic masks, DINOv2 ViT-S/14 CLS
descriptors, masked crops, one fixed reference, cosine matching. Query GT was
loaded only after proposal generation and matching.

| Dataset | Queries | Oracle proposal recall | Matching accuracy given oracle | Recall@1 | Recall@3 | Mean selected mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| YCBInEOAT `cracker_box_reorient` | 3 | 100% | 66.67% | 66.67% | 100% | 0.594 |
| HOT3D clip 001852, object 29 | 3 | 100% | 100% | 100% | 100% | 0.859 |

Mean end-to-end runtime on RTX 4060 was 4.29 s/query for 640×480 YCBInEOAT
(SAM 3.80 s, matching 0.45 s) and 45.97 s/query for 1408×1408 HOT3D (SAM
39.49 s, matching 6.19 s). Full-resolution SAM automatic mask generation is
therefore the dominant HOT3D bottleneck.

The sample is intentionally small and is an integration/pilot result, not a
paper-level estimate. YCBInEOAT contains one matching failure: SAM produced a
valid target proposal, but DINOv2 ranked a distractor/partial region first.
HOT3D succeeded on all three deterministic sampled frames. These results
justify proceeding to a larger validation split, but do not yet establish
cross-sequence generalization.

Full rankings, Top-5 masks, per-frame timings, and all debug images remain in
the gitignored `outputs/ycbineoat_reference_init_v1/` and
`outputs/hot3d_reference_init_v1/` directories.
