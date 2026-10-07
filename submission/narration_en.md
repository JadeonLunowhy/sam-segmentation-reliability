# Presentation script

Target: 4-4.5 minutes. Rehearse and adjust speaking pace. Allocate all sections among actual students; every member must speak.

## Slide 1: Research question

Our project asks when a frozen segmentation model can recognize its own failures. We use the original Segment Anything Model, or SAM, without training or fine-tuning. The aim is to rank unreliable masks for review. We do not alter the mask itself. Our strongest counterargument is simple: a model can produce the same wrong segmentation repeatedly. We test that possibility instead of assuming stability means correctness.

## Slide 2: Related work

SAM predicts its own mask quality. PointPrompt studies the role of prompts. Prompt perturbation uncertainty also already exists, including a recent medical-image study with largely negative comparisons. We therefore do not claim a new uncertainty principle. Our contribution is a controlled natural-image analysis of click position, perturbation radius, compute budget, and a separate public image corpus.

## Slide 3: Protocol

We split the main dataset by image: forty-five validation images and one hundred and six test images. We separately download seventy-four Pet images, two per breed. Each image has an interior and a boundary click simulated from annotations. Labels are used only for simulation and evaluation. All scoring code receives the image, click and predictions. Uncertain annotation borders are ignored, and every image's two clicks stay together in statistical resampling.

## Slide 4: Methods

The baselines are SAM confidence, threshold stability and horizontal-flip consistency. Our prompt score averages overlap between the original mask and nearby-click masks. We also test minimum overlap and equal-weight fusion with confidence. We sweep three radii and three budgets. Validation chooses r0.06_k8; that choice is frozen before test evaluation. Nearby clicks may leave the object, so we record that diagnostic instead of secretly fixing prompts with labels.

## Slide 5: Main results

On the main test set the original masks have mean IoU 0.475. Confidence has AURC 0.430, and fusion has AURC 0.333. Lower is better. The paired difference is -0.0971. The ninety-five percent interval runs from -0.1449 to -0.0528. It excludes zero within this evaluation protocol. The curves show whether refusing low-confidence cases actually lowers retained error.

## Slide 6: Additional corpus

We apply exactly the same configuration to the Pet corpus. Mean IoU is 0.812; confidence AURC is 0.103, compared with fusion AURC 0.051. This checks sensitivity to a second set of natural images. It does not establish that those images were absent from SAM pretraining. We separately inspect radius and budget sensitivity, and report subgroup results as exploratory.

## Slide 7: Stable failures

We define a stable failure in advance as prompt consistency at least point nine and true IoU below point five. There are 2 such main-test cases and 0 Pet cases. The examples show the original image, the reference object and the prediction. These cases reveal the fundamental limitation: agreement can describe a repeated mistake. A reliable screening method must be tested against labels, even when it does not use labels at inference.

## Slide 8: Compute

Prompt variations reuse one cached image embedding, while horizontal flip needs another encoder pass. We measure actual batches of two, four and eight decodes on five validation images, rather than assuming cost scales linearly. The full experimental sweep costs more than deploying just the chosen method. Quality and cost must be read together: cheap inference is useful only if its ranking is informative.

## Slide 9: Conclusion

The project provides executable code, preserved observations, controlled comparisons and image-level bootstrap intervals, with no training. Its conclusion follows the measured results, including negative findings. Limits include simulated users, one backbone and finite datasets. Future work should test real user clicks and additional models. Each team member should present their assigned section and document their actual contribution.