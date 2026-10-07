# Training-Free Reliability Screening for Prompted Segmentation

Design date: 2026-10-03 (Asia/Shanghai). Status: design and implementation plan approved by the user; technical implementation and actual experiments complete. See verification.md and requirements_matrix.md for evidence and remaining human inputs.

## Objective and contribution

Investigate whether prompt-response consistency predicts failures of a frozen Segment Anything Model (SAM) on natural images, and when stable predictions remain wrong. The output is a reproducible research project, not a claim of a new segmentation backbone or of first inventing prompt perturbation.

Primary question: At what additional inference cost can prompt perturbations improve failure screening over SAM's predicted IoU and threshold stability, across different click locations and datasets?

Hypotheses to test, not conclusions:

1. Prompt consistency may improve failure ranking for ambiguous clicks.
2. Confidence and consistency contain partially complementary information.
3. Consistency can miss masks that are stable but wrong.
4. Additional decodes using a cached image embedding may be cheaper than image transformation consistency, without necessarily being more accurate.

The course contribution is an independently implemented controlled natural-image study, including click placement, perturbation budget and radius, paired uncertainty estimates, stable-error analysis and a separate additional image corpus. Any score improvement must be supported by actual results. Negative results remain in the report.

## Closely related work and limits of novelty

- Kirillov et al., Segment Anything, ICCV 2023: https://arxiv.org/abs/2304.02643 ; official code https://github.com/facebookresearch/segment-anything
- Quesada et al., PointPrompt, CVPR Workshops 2024: https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Quesada_PointPrompt_A_Multi-modal_Prompting_Dataset_for_Segment_Anything_Model_CVPRW_2024_paper.html
- Kaiser, Norrenbrock and Rosenhahn, UncertainSAM, ICML 2025: https://proceedings.mlr.press/v267/kaiser25a.html . This method trains a lightweight uncertainty model, so it is related work, not a training-free implementation baseline.
- Loke et al., When Does Prompt-Perturbation Uncertainty Catch Interactive-Segmentation Failures?, MICCAI satellite open-access 2026: https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html . Directly related empirical work in medical images; its reported negative results motivate careful paired comparisons. Do not claim prompt perturbation is new.

No literature search can establish universal novelty. The report must describe this as a course-level empirical contribution and explicitly acknowledge prior perturbation studies.

## Model and environment

- Use official SAM ViT-B public pretrained weights; inference only, model.eval(), all parameters frozen, torch.inference_mode().
- Select the original mask among SAM's three candidates using highest predicted IoU, never ground-truth overlap.
- Use the same selection rule for every perturbed prediction.
- Keep the original segmentation unchanged. The research compares reliability scores, not repaired masks.
- Install dependencies only in project/.venv. Record exact versions and installation commands in docs/environment.md and requirements-lock.txt. Do not alter system Python or GPU drivers.
- CPU is the verified fallback. GPU use requires a compatible wheel for the RTX 5060 laptop GPU and a real successful inference check; no unsupported CUDA assumptions.
- Record model hash, upstream source version, device, image encoding and mask decoding time, peak memory where measurable, and the number of model calls.
- Preserve DLCV Project Overview.pdf byte-for-byte.

## Data

Main benchmark: Oxford Interactive Image Segmentation dataset, 151 images with ground-truth segmentations, approximately 22 MB of images. Official source: https://robots.ox.ac.uk/~vgg/data/iseg/ . Its source composition includes GrabCut, PASCAL VOC and Alpha matting images; disclose this and avoid describing them as newly photographed images.

Use a seeded image-level split of 45 validation and 106 held-out test images, subject to integrity checks of the actual archives. A corrupt or unusable item is logged before inference, never removed because the method fails.

Additional corpus: a seeded selection of 74 Oxford-IIIT Pet images from the official test split, two per breed where available. Official source: https://www.robots.ox.ac.uk/~vgg/data/pets/ . Treat this as separately internet-collected public data, not original photography or a newly created dataset. Oxford Pet trimaps use foreground, background and an uncertain border; ignore the uncertain border for IoU and disclose the convention.

Save raw URLs, hashes, image identities, labels, extraction rules, sources, exclusion logs and fixed splits. Check for exact duplicate images across sources. If downloads fail, document the failure and choose a comparable officially annotated natural-image corpus before freezing the experiment; never replace real evaluation with synthetic numbers.

The assignment's extra-data wording is interpreted as allowing separately downloaded additional public images. This interpretation is disclosed; the project does not claim personally collected photographs.

## Prompt protocol and leakage controls

For each image, simulate two positive-click regimes:

1. Interior click at the foreground distance-transform maximum.
2. Seeded foreground click near the object's boundary, at a fixed image-relative boundary distance band; log a deterministic fallback when the band is empty.

Ground truth is used only to simulate the target click and evaluate predictions. These are controlled simulated users, not a real human-user study. The reliability scorer receives image, prompt and predicted masks only, never annotation data.

Perturb the point in eight fixed angular directions. Sweep radii 1%, 3% and 6% of image diagonal; evaluate K=2,4,8 decodes using fixed subsets of directions. Clip to image coordinates. Do not use ground truth to silently move perturbed clicks back into the object. Report the fraction of perturbations leaving the target as a diagnostic and repeat the analysis on cases where they remain inside. This diagnostic cannot affect deployed scoring.

Split by image before prompt creation. All clicks and perturbations of one image belong to the same split. Freeze method configuration after validation; do not choose a radius, fusion rule or failure threshold using held-out results.

## Reliability scores and baselines

All scores have the convention larger = more reliable.

1. SAM predicted IoU of the selected original candidate; this is a self-rating, not true IoU.
2. Official threshold stability, calculated from mask logits at threshold +/- 1 using the upstream definition.
3. Prompt consistency: mean IoU between the original prediction and K perturbed predictions.
4. Minimum prompt consistency: worst pair overlap with the original, as an ablation.
5. Fixed equal-weight fusion: (clip(predicted_iou,0,1) + mean_consistency)/2. No learned weights or classifier.
6. Horizontal-flip image consistency: transform image and click, run SAM, map the predicted mask back and compare with the original. Disclose the extra image encoder pass.
7. Random rejection expectation and annotation-based oracle ordering, clearly identified as reference bounds rather than usable methods.

Select the perturbation radius and K by validation AURC for the fixed fusion rule, with lower compute as the tie-breaker. Save the frozen configuration before test inference/evaluation. Test tables may include the preregistered sweep as sensitivity analysis, but cannot redefine the primary result using the best test cell.

Mask IoU(empty,empty)=1 for agreement scores; true target masks must be nonempty, so an empty predicted mask receives true IoU=0. Test this explicitly.

## Evaluation and statistical analysis

- Primary selective risk: 1 - true IoU; plot mean retained risk versus retained fraction, and calculate AURC using a specified finite-sample convention.
- Failure definitions: true IoU below 0.5 (primary) and below 0.75 (secondary).
- Report failure-detection AUROC and average precision with risk scores = 1 - reliability, failure counts, and prevalence. A one-class subset returns undefined, not an invented 0.5.
- Report mean true IoU after retaining 90%, 75% and 50% of cases, plus the rejected failure fraction.
- Report score/IoU rank association and compute cost.
- Use paired bootstrap resampling by image, retaining both prompt regimes together, 2,000 draws, fixed seed; report confidence intervals for score differences. Keep extra-corpus estimates separate.
- Resolve ties in ranking consistently; do not reward tied scores based on ground truth.
- Break down results by prompt regime, object area, model confidence and data source; label subgroup analysis exploratory.
- Define stable-error examples in advance as mean consistency >= 0.9 and true IoU < 0.5. Include examples when present; explicitly report if no such cases occur.
- Save per-image/per-prompt observations, mask outputs for reproducible audit, full timing logs and all figure source data.

## Deliverables

- README.md: Chinese explanation, English project title, setup/reproduction commands and verified completion status.
- src/: data loading, prompt creation, frozen SAM inference, reliability scores, evaluation and image-level bootstrap.
- scripts/: reproducible downloads, experiment runner, analysis, demo and artifact builders.
- configs/: preregistered study configuration and frozen validation selection.
- tests/: meaningful scientific-correctness tests for IoU conventions, stability, score direction, ties, one-class metrics, prompt coordinate mapping, image-level split isolation and grouped bootstrap.
- data/: raw annotated corpora, provenance and split manifest.
- models/: public frozen checkpoint and hash.
- results/: actual per-case data, validation selection, test summaries, bootstrap intervals and compute costs.
- figures/: actual plots and qualitative examples, with source tables.
- submission/proposal.pdf: full sentences, one single-spaced page, 2-4 relevant papers, data, method, compute, evaluation and expected impact.
- submission/report.pdf plus editable source: approximately five pages excluding references, all required paper sections and individual contributions. All numbers come from saved actual results.
- submission/presentation.pptx: editable English research deck with inspected rendering.
- submission/narration_en.md and narration_zh.md: approximately 4-4.5 minute speaking script and explanation, split among actual members when supplied.
- submission/preview_video.mp4: timed slide preview clearly labeled as a rehearsal aid; it cannot satisfy the requirement that every actual student presents.
- docs/requirements_matrix.md, environment.md and verification.md: requirement coverage, installed-package record and actual checks.

## Acceptance and outstanding human inputs

Technical acceptance: model performs real inference; meaningful tests pass; full configured validation, test and additional-corpus experiments execute; outputs can be rebuilt from saved observations; artifacts render without clipped content; README accurately states any failures or unfinished parts.

No fabrication of positive results, contributions, names, student IDs or work hours. Team size, identity and actual individual contributions remain unknown. All-member speaking and at least 15 hours of real work per member cannot be completed or certified by an automated slide preview. No iSpace submission is authorized by this task.

Team-size scope must be finalized after team size is supplied. The proposed multi-axis analysis and cross-corpus evaluation provide a substantive starting scope; do not promise a course grade.

## Implementation stages after design review

1. Create a detailed implementation plan and verify dependency/data download feasibility.
2. Implement and test data, scoring and evaluation; run a real SAM inference smoke test.
3. Run validation, freeze configuration, execute held-out and additional-corpus experiments.
4. Audit results, create figures and write conclusions supported by actual evidence.
5. Build, render and inspect proposal, report, presentation and rehearsal video.
6. Run final reproducibility checks and deliver an honest requirement matrix.
