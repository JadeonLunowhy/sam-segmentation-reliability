# Final verification

Completed locally on 2026-10-04, Windows 11, Python 3.13.4, PyTorch 2.8.0+cu128 and RTX 5060 Laptop GPU. All Python installations are in project/.venv. Exact dependencies and model/source/data provenance are preserved.

## Scientific execution

- Final full run: 45 validation images / 90 cases, 106 main-test images / 212 cases, 74 additional Pet images / 148 cases. No exclusions or duplicated decoded images. Exact image-level split and per-file image/target/valid SHA-256 fingerprints are in data/splits.json.
- All parameters frozen in each split's actual runtime metadata. Two image encodings per image (original and flip), 52 prompt decodes per image across two regimes and the registered sweep. Final totals: 450 encodings and 11,700 prompt predictions, with perturbations batched. These are prompt counts, not counts of batched forward calls.
- Validation-only selection remains r0.06_k8. Frozen.json binds validation observations, model, data manifest and study configuration. Held-out data did not select parameters.
- Independent numerical reviewer recomputed all initial 450 IoUs and all score cells with maximum difference zero. Two implementation issues were corrected with failing/passing regression tests, then all experiments were rerun. Final numerical values remained identical. Review details and its interrupted final verdict are documented in review.md.
- scripts.audit verifies all final input and mask hashes, dimensions, unique prompt cases, score ranges and recomputes each saved IoU against reference annotations. results/audit.json records actual outcomes.
- Summary and all bootstrap outputs regenerate byte-for-byte from final saved observations: results/rebuild_verification.json. Bootstrap resamples images, keeping both clicks together.
- Main confidence/fusion AURC: 0.4296536642 / 0.3326001184; paired difference -0.0970535458, 95% interval [-0.1449289074, -0.0528121883]. Pet: 0.1025590345 / 0.0513865421, difference -0.0511724924, interval [-0.0840272604, -0.0239693080]. Mean consistency alone is slightly better than fusion on main AURC; fusion is not claimed superior to it.
- Real annotation-free image-and-click demo passed. No reference-mask input is accepted by the demo scorer. Scores are rankings, not calibrated correctness probabilities.

## Tests and artifacts

- 21 unit tests pass; actual command/output in tmp/tests-final.log. Regression tests cover input-file tampering and editable-PDF-source preservation. pip check reports no broken requirements.
- Proposal: 1 page. Report: 5 body pages plus 1 reference page. Every final PDF page rendered with pdftoppm and visually inspected; actual mask overlays are aligned, captions readable, no content clipping.
- Deck: 9 slides, all individually rendered and inspected. Editable native tables on slides 3/4 and a native timing chart with embedded data workbook on slide 8. Consecutive rebuilds pass package, geometry, fonts, table/chart and first-party import checks. Authoring uses existing Artifact Tool runtime, not an additional system installation. results/presentation_validation.json binds the delivered PPTX bytes.
- Silent rehearsal video: target and observed duration 260 seconds, 1280x720, H.264; complete ffmpeg decode verifies playability. results/video_validation.json records the actual check. This is not an all-member speaking recording.
- Nonempty editable Markdown sources include report prose, measured tables and figure links. English and matching Chinese scripts are present; actual speaker assignment awaits real team information.
- Original assignment SHA-256 remains 423cf3e7139981d984d88f13afe97e751d0e7fd6f1b639ddbda58a1f35c35e7d.

## Human items still required

Real student names/IDs, verified individual contributions and actual hours, team-size workload confirmation, and each member's personally recorded speaking segment. These are intentionally pending in team.json and the requirement matrix. No submission to the course portal was performed.
