# Segmentation Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native execution in the current session is recommended; do not delegate implementation unless the user chooses it.

**Goal:** Complete a real, reproducible, training-free SAM reliability study and its course submission materials in this folder.

**Architecture:** Separate annotated data/prompt simulation from a label-blind frozen-model scorer. Cache each image embedding, save all per-prompt observations, choose the primary perturbation configuration on validation only, and derive all tables and artifacts from held-out observations.

**Tech Stack:** Project-local Python virtual environment; PyTorch, official SAM source, NumPy, Pillow, SciPy, Matplotlib and metric utilities; PDF tools; the presentation skill's JavaScript artifact tooling when available; a project-local video encoder when needed. Pin actual installed versions after compatibility checks rather than inventing a lock file in advance.

**Spec:** [Approved design](design.md)

## Global Constraints

- Use official SAM ViT-B public pretrained weights; inference only, model.eval(), all parameters frozen, torch.inference_mode().
- Install dependencies only in project/.venv. Record exact versions and installation commands in docs/environment.md and requirements-lock.txt. Do not alter system Python or GPU drivers.
- Preserve DLCV Project Overview.pdf byte-for-byte.
- Use a seeded image-level split of 45 validation and 106 held-out test images, subject to integrity checks of the actual archives.
- Additional corpus: a seeded selection of 74 Oxford-IIIT Pet images from the official test split, two per breed where available.
- Perturb the point in eight fixed angular directions. Sweep radii 1%, 3% and 6% of image diagonal; evaluate K=2,4,8 decodes using fixed subsets of directions.
- Save the frozen configuration before test inference/evaluation.
- Use paired bootstrap resampling by image, retaining both prompt regimes together, 2,000 draws, fixed seed; report confidence intervals for score differences.
- No fabrication of positive results, contributions, names, student IDs or work hours.
- Keep experiments, downloaded weights, dependencies and generated materials within this project. No external publication or course-system submission.

Additional fixed conventions: seed=20261003; boundary band is distance to foreground edge in (0, 0.02 * image diagonal], with the distance-transform maximum as a logged fallback; perturbation directions are multiples of pi/4 with K subsets [0,4], [0,2,4,6] and [0,1,2,3,4,5,6,7]. These opposite/balanced subsets avoid directional bias.

## Review Focus

1. Thin targets or an empty boundary band: choose a valid positive point and log the fallback; never send background as a nominal positive click.
2. Empty predictions, ignored annotation borders or zero valid union: use explicit IoU conventions and never turn missing annotation into a correct prediction.
3. Equal reliability scores or one-class failure labels: use tie-aware metrics and return undefined where the metric cannot be estimated.
4. Image edge clicks, horizontal flips and resized images: coordinates and masks must map back correctly without clipping into the wrong axis.
5. Interrupted downloads/runs and repeated prompts: resume only validated completed items; never mix splits or treat two prompts from one image as independent bootstrap samples.

The existing folder is not a Git checkout. Work directly here, as requested; do not manufacture branches or commit steps. Record actual checks in docs/verification.md.

---

### Task 1: Reproducible environment and annotated data

**Files:** Create scripts/setup.ps1, scripts/download_data.py, src/data.py, configs/study.json, data/provenance.json, data/splits.json, docs/environment.md, requirements-lock.txt, tests/test_data.py and .gitignore.

**Interfaces:**
- `Case(image_id: str, dataset: str, split: str, image_path: Path, target_path: Path, valid_path: Path)` records one image's prepared evaluation inputs.
- `prepare_data(config: dict) -> list[Case]` verifies downloads, writes prepared masks and split manifest.
- `load_case(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returns RGB uint8 image, bool foreground, bool valid pixels in original dimensions.
- `sha256_file(path: Path) -> str` streams the file hash.

- [ ] **Step 1:** Inspect official download availability and Python/PyTorch compatibility. Create .venv, install only local dependencies, record commands and perform a device tensor smoke test. Prefer a compatible GPU build if practical; retain the verified CPU fallback.
- [ ] **Step 2:** Write and run failing tests for split disjointness, deterministic selection and trimap conversion: labels [1,2,3] must produce foreground [True,False,False] and valid [True,True,False]. Include mismatched image/mask dimensions, incomplete download and unsafe archive paths.
- [ ] **Step 3:** Implement safe downloads with temporary files and atomic completion, archive extraction restricted to project/data/raw, source hashes and data adapters. Inspect actual interactive-dataset label semantics before mapping; do not guess from filenames or silently binarize uncertain labels.
- [ ] **Step 4:** Download the main corpus and the separate Pet corpus, verify nonempty target masks, construct fixed splits and check exact decoded-image duplicates. Write exclusions before inference, with reason.
- [ ] **Step 5:** Run `.\.venv\Scripts\python.exe -m unittest tests.test_data -v`; require passing tests and a manifest with 151 usable main images and 74 extra images, or explicitly document an archive-driven scope revision.

### Task 2: Prompt simulation and ground-truth-free reliability scores

**Files:** Create src/prompts.py, src/scoring.py, tests/test_prompts.py and tests/test_scoring.py.

**Interfaces:**
- `make_prompt(target: np.ndarray, regime: str, seed: int) -> tuple[np.ndarray, dict]` returns xy coordinate and simulation diagnostics.
- `perturb_points(point: np.ndarray, shape: tuple[int,int], radius_fraction: float) -> np.ndarray` returns eight xy coordinates; no mask argument.
- `flip_point(point: np.ndarray, width: int) -> np.ndarray` uses x'=width-1-x.
- `mask_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float` returns agreement with the empty-union convention.
- `true_iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float` validates a nonempty annotated target.
- `threshold_stability(logits: np.ndarray, threshold: float = 0, offset: float = 1) -> float` matches official SAM behavior, including a documented empty denominator.
- `reliability_scores(predicted_iou: float, original: np.ndarray, perturbations: np.ndarray, flipped: np.ndarray) -> dict` consumes predictions only.

- [ ] **Step 1:** Write failing tests: identical masks give IoU=1; disjoint nonempty masks give 0; empty-vs-empty agreement gives 1; empty prediction against foreground gives true IoU=0; ignored border pixels do not affect true IoU. Verify logits [-2,0,2] give threshold stability 1/2 at offset=1.
- [ ] **Step 2:** Write failing prompt tests for thin masks, invalid empty target, points near all four image edges, opposite-direction K subsets, flip round-trip and deterministic seeded boundary clicks. Verify score functions accept no target mask.
- [ ] **Step 3:** Run tests to observe meaningful failures, implement the interfaces, then rerun both suites until they pass.
- [ ] **Step 4:** Save prompt simulation metadata separately from score inputs. Mark ground-truth-based perturbation validity as evaluation-only metadata, not a filtering rule.

### Task 3: Frozen SAM inference, resumable observations and demo

**Files:** Create src/inference.py, scripts/run_experiment.py, scripts/demo.py, models/provenance.json and tests/test_inference.py.

**Interfaces:**
- `SamRunner(checkpoint: Path, device: str)` loads official ViT-B and exposes `encode(image: np.ndarray) -> dict` and `predict(point_xy: np.ndarray) -> Prediction`.
- `Prediction(mask: np.ndarray, logits: np.ndarray, predicted_iou: float, decode_seconds: float)` includes the selected original-resolution candidate.
- `run_split(split: str, config_path: Path, resume: bool = True) -> Path` saves validated observations and required masks; refusal to run test without frozen validation config is mandatory.
- Every observation contains image ID, split, dataset, click regime/xy, true IoU, score variants, perturbation metadata, encoder/decoder costs, call counts, model/config hashes and mask path.

- [ ] **Step 1:** Write failing tests with a fake predictor: highest predicted IoU selects the mask even when another candidate matches ground truth; all parameters remain frozen; two original clicks reuse one encoding; flip consistency uses an additional encoding; incomplete rows cannot be resumed as completed cases.
- [ ] **Step 2:** Implement the runner under inference_mode, CUDA timing synchronization where applicable, atomic observation writes and hash-checked resumability. Store the official source revision and checkpoint hash.
- [ ] **Step 3:** Run a real one-image smoke test. Save original RGB, prompt, prediction and ground truth overlay for visual inspection; verify dimensions, coordinate convention and annotation labels.
- [ ] **Step 4:** Run `scripts.demo` on an image/point input without an annotation file. Verify it emits segmentation, predicted confidence and perturbation reliability, and does not claim reliability is a calibrated probability.
- [ ] **Step 5:** Run inference tests and save a smoke-test record containing actual model/device and timings. No training, gradients, optimizer or checkpoint updates may occur.

### Task 4: Tie-aware evaluation and paired image bootstrap

**Files:** Create src/metrics.py, src/statistics.py, scripts/analyze.py, tests/test_metrics.py and tests/test_statistics.py.

**Interfaces:**
- `selective_curve(iou: np.ndarray, reliability: np.ndarray) -> tuple[np.ndarray,np.ndarray]` returns coverage and retained risk; tied score groups use expected risk for a random ordering within the group, independent of ground truth ordering.
- `aurc(iou: np.ndarray, reliability: np.ndarray) -> float` uses mean retained risk at coverages k/n, k=1..n.
- `failure_metrics(iou: np.ndarray, reliability: np.ndarray, threshold: float) -> dict` returns AUROC, average precision, prevalence and counts; encode undefined results as JSON null.
- `bootstrap_difference(rows: list[dict], first: str, second: str, metric: str, draws: int = 2000, seed: int = 20261003) -> dict` resamples image IDs and includes their prompt regimes together.
- `freeze_selection(validation_path: Path, config_path: Path) -> Path` minimizes fusion AURC and breaks ties by K then radius, writing configs/frozen.json with input hashes.

- [ ] **Step 1:** Write failing metric tests: for IoU [1,0] and reliability [1,0], AURC=0.25; reversing reliability gives 0.75; tied reliability gives 0.5 regardless of row permutation. Perfect failure ranking gives AUROC=1; all-success or all-failure subsets have undefined AUROC.
- [ ] **Step 2:** Write failing bootstrap tests: rows for both regimes of one image are sampled together; identical score columns give paired AURC difference=0 in every draw; fixed seed reproduces intervals. Ensure duplicate bootstrap copies retain multiplicity.
- [ ] **Step 3:** Implement metric and selection interfaces, run the tests, and explicitly test reliability-to-risk direction and both failure thresholds.
- [ ] **Step 4:** Test that modifying held-out rows cannot alter validation selection, and that frozen manifest hashes detect a changed validation input.

### Task 5: Actual experiments, scientific audit and figures

**Files:** Create results/validation.jsonl, results/test.jsonl, results/extra.jsonl, results/summary.json, results/bootstrap.json, figures/ and docs/results_audit.md.

**Interfaces:** Consumes Tasks 1-4. Produces immutable observations, source tables and evidence figures for Task 6. The main test set and additional corpus are reported separately.

- [ ] **Step 1:** Execute validation on the planned 45 images and both prompt regimes. Inspect model outputs for label/coordinate errors; fixes to correctness must be documented and validation regenerated.
- [ ] **Step 2:** Freeze radius/K by the approved validation rule and persist config and data hashes before held-out inference. No learned fusion weights.
- [ ] **Step 3:** Execute all 106 main test images and 74 extra images with both prompt regimes. Save the registered sweep for sensitivity analysis; the primary method remains the validation selection. Resume interrupted cases without silently reducing sample count.
- [ ] **Step 4:** Analyze primary and secondary failure definitions, prompt regimes, object size, target-leaving perturbations, stable errors and inference costs. Use paired image bootstrap for main method-vs-confidence AURC and AUROC differences; label remaining subgroup results exploratory.
- [ ] **Step 5:** Generate selective-risk curves, method comparison tables, radius/K sensitivity, compute-versus-quality plot and qualitative success/failure examples from saved data. Save each figure's source CSV/JSON. Inspect all figures for readable axes and correctly aligned overlays.
- [ ] **Step 6:** Audit counts, splits, score ranges, undefined metrics, actual model call counts and ground-truth leakage. Report null or negative findings as observed; do not claim significance from a confidence interval crossing zero.

### Task 6: Submission materials and final verification

**Files:** Create README.md, submission/proposal.pdf, submission/report.pdf, editable report/proposal sources, submission/presentation.pptx, submission/narration_en.md, submission/narration_zh.md, submission/preview_video.mp4, submission/team.json, scripts/build_materials.*, docs/verification.md; update docs/requirements_matrix.md.

**Interfaces:** Read results/summary.json, bootstrap.json, source tables and verified figures. Materials builders must fail if referenced actual result files are absent. The preview video is a rehearsal aid, not a substitute for student presentation.

- [ ] **Step 1:** Apply PDF and presentation skill instructions and inspect available authoring/rendering tools. Follow their artifact-start marker contract when the required helper exists; record any missing runtime capability and use an available local fallback transparently.
- [ ] **Step 2:** Write a one-page full-sentence proposal with 2-4 relevant papers, and an approximately five-page report excluding references covering abstract, introduction, related work, method, results, discussion, conclusion and individual contributions. Reference actual selected configuration and actual figures; distinguish simulated clicks, additional public images and real human inputs.
- [ ] **Step 3:** Build an editable English deck of approximately 8-10 slides, with actual data evidence, an accurate conclusion and limitations. Add matching English and Chinese speaking scripts with a target of 4-4.5 minutes and explicit speaker assignment instructions.
- [ ] **Step 4:** Render every report/proposal page and every slide, visually inspect, correct clipping/overlaps and verify report length. Create a timed rehearsal MP4 from inspected slide renders; verify its duration and playability. Clearly label its role in README and its opening frame.
- [ ] **Step 5:** Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`, the annotation-free demo, and analysis regeneration from saved observations. Compare regenerated summaries to saved ones. Verify package lock, checkpoint/source/data hashes and unchanged original assignment PDF.
- [ ] **Step 6:** Record only checks actually completed. Update requirement statuses and provide links to final artifacts. Leave actual identities/contributions/hours and all-member recording pending when not supplied, rather than presenting a technically complete package as fully ready for submission.

## Plan self-review

Coverage: all approved model, data, prompt, score, statistical and deliverable requirements map to Tasks 1-6. No training is introduced. Every review-focus item has a targeted test or artifact inspection. Scorer inputs exclude annotations; validation selection is separated from test evaluation; bootstrap is grouped by image. Installation stays local. No existing Git integration is required.

Outstanding decisions requiring real input are team size, student identities and actual contributions. These do not block technical implementation. Model/device compatibility, archive labels and authoring-tool availability require actual checks in their owning tasks, not guessed declarations.

Status: approved and technically implemented. See docs/progress.md and docs/verification.md for actual checks, replacement checks and remaining human inputs. Checkbox descriptions are the original plan; real completion evidence is recorded separately.
