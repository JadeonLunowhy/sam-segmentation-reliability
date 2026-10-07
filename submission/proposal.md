Training-Free Reliability Screening for Prompted Segmentation

Research question and motivation

A plausible segmentation can be wrong while appearing stable and receiving a high model confidence score. We investigate whether small perturbations of a user click provide useful evidence for screening failures of a frozen Segment Anything Model (SAM), and when consistency misses stable errors. The goal is to identify unreliable outputs, not to train a new segmenter or guarantee accurate masks.

Prior work and scope

SAM provides promptable segmentation and predicted mask quality [1]. PointPrompt studies prompting behavior [2]. UncertainSAM learns a lightweight uncertainty estimator [3], whereas this project updates no weights. A recent prompt-perturbation study reports mixed or negative results on medical data [4]. Our contribution is a controlled natural-image study of click placement, perturbation radius, inference budget and transfer to a separately collected public corpus; prompt perturbation itself is not claimed as new.

Data, method and resources

We use 151 images from the Oxford interactive segmentation benchmark, with 45 image-level validation cases and 106 held-out images. An additional internet-downloaded corpus contains 74 Oxford-IIIT Pet test images, two per breed, using official trimaps. Uncertain annotation borders are ignored. For each image, we simulate an interior click and a near-boundary click; these use annotations only to define the target and evaluate output. Official SAM ViT-B weights remain frozen. The scorer compares the original prediction with masks from two, four or eight perturbed clicks at radii of one, three or six percent of the image diagonal. Fixed fusion averages consistency and clipped SAM confidence. A horizontal-flip baseline requires an extra image encoding. Computation uses the verified local device, without any model training.

Evaluation and expected contribution

We compare SAM confidence, threshold stability, mean and minimum prompt consistency, fixed fusion and flip consistency. Validation selective risk selects the primary radius and budget before test evaluation. We report failure-detection AUROC and average precision, selective risk curves, AURC, retained IoU, inference costs and paired image-bootstrap intervals. Failure thresholds are IoU below 0.5 and 0.75. Perturbations leaving the target are analyzed explicitly rather than filtered using labels. We expect interpretable plots, error examples and evidence about whether extra inference is worthwhile; a negative finding is a valid outcome. If useful, the resulting scoring code can flag masks for human review without retraining.

References

[1] A. Kirillov et al. Segment Anything. ICCV, 2023. https://arxiv.org/abs/2304.02643

[2] J. Quesada, M. Alotaibi, M. Prabhushankar and G. AlRegib. PointPrompt: A Multi-modal Prompting Dataset for Segment Anything Model. CVPR Workshops, 2024. https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Quesada_PointPrompt_A_Multi-modal_Prompting_Dataset_for_Segment_Anything_Model_CVPRW_2024_paper.html

[3] T. Kaiser, T. Norrenbrock and B. Rosenhahn. UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model. ICML, PMLR 267, pp. 28670-28688, 2025. https://proceedings.mlr.press/v267/kaiser25a.html

[4] S. Loke et al. When Does Prompt-Perturbation Uncertainty Catch Interactive-Segmentation Failures? MICCAI satellite open-access paper, UNSURE, 2026. https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html