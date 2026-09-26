# Motion Proposal Failure Analysis

## Current status

The phase-1 implementation has passed deterministic synthetic sanity tests for
static scenes, camera translation, independently moving rectangles, combined
camera/object motion, border expansion, small strong components, temporal
persistence, and proposal metrics. These tests establish code and mathematical
sanity only; they are not evidence of recall on natural continuous video.

One qualitative smoke run used the public OpenCV
[`vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi)
sample for 30 frame pairs. It has no target-specific GT in this project, so no
Recall@K is reported.
The homography quality gate passed 30/30 pairs; mean inlier ratio was 0.969,
median-inlier reprojection error averaged 0.031 pixels, and CPU runtime averaged
0.193 s/pair on the local host. The configured Top-10 output averaged 9.67
proposals/frame. Visual inspection showed proposals on the moving pedestrians,
but also fragmented/background candidates. This is useful pipeline evidence and
also shows that the untuned defaults are recall-oriented and too permissive for
any precision claim.

The currently downloaded BOP19 YCB-V subset contains sparse target frames, so
it is not used as a motion-proposal benchmark. No natural target-specific
benchmark result is reported yet; the rest of this document records the
collection and diagnostic protocol rather than inventing observations.

## Automatic failure collection protocol

After running a genuinely continuous validation sequence, join evaluator output,
`frames.csv`, `proposals.csv`, and dataset attributes by sequence/frame/object.
Collect four non-exclusive categories:

1. **False negative:** no proposal reaches evaluator IoU, plus center-recall
   status for small targets.
2. **False positive load:** high proposal count or high-scoring regions not
   overlapping the target. A non-target moving object is not a flow error; it is
   expected input for later reference verification.
3. **Fragmentation:** several proposals overlap different parts of one GT box.
4. **Over-merge:** one proposal spans multiple independently moving instances.

For every collected frame, retain:

- original RGB pair and six-panel debug image;
- proposal and GT overlays, with GT marked evaluator-only;
- background model type, validity, inlier ratio, coverage, reprojection error;
- raw/background/residual flow statistics and adaptive threshold;
- target visibility, scale, motion, sharpness, and clutter attributes when known;
- all proposal score components and tracklet history;
- config, seed, software version, and runtime.

## Failure hypotheses to test

| Symptom | Candidate causes | Evidence to inspect |
|---|---|---|
| False negative | static/slow relative target, camera-motion cancellation, tiny target, blur, occlusion, flow failure | target residual distribution, scale, sharpness, center recall |
| Background leakage | parallax, weak texture, bad RANSAC coverage, dynamic background | model validity, inlier map coverage, residual heatmap |
| Fragmentation | texture gaps, partial motion, morphology too weak, occlusion | component masks, flow direction, merge decisions |
| Over-merge | nearby movers, expansion too large, permissive centroid merge | pre/post-merge boxes and direction cosine |
| Temporal flicker | noisy flow, reflection, association threshold | tracklet hits/misses, IoU and centroid trajectory |

## Required evidence before an upgrade

First compare raw flow, affine compensation, homography compensation, and the
full pipeline on validation video. Upgrade to RAFT/GMFlow only if Farneback flow
is a measured bottleneck. Upgrade the background model to piecewise affine,
multiple homographies, or epipolar residual only if valid-homography diagnostics
show parallax is a dominant recall failure. Do not add SLAM or depth merely from
qualitative suspicion.

The phase-1 decision question remains:

> Does global-motion-compensated residual optical flow provide sufficiently
> high-recall full-frame proposals at an acceptable proposal count and runtime?
