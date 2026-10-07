# Independent review and fixes

A fresh, read-only reviewer inspected the scientific implementation and actual observations. Its numerical audit independently recomputed all 450 IoUs and every radius/K consistency, minimum, fusion and flip score: maximum difference zero. It independently selected r0.06_k8 and reproduced all four primary confidence/fusion AURCs. Current original-image hashes matched the manifest; official SAM source revision matched provenance and the source checkout was clean.

Two Important findings were returned:

1. Editable Markdown sources were empty because ReportLab consumes its input story. Regression `test_pdf_export_preserves_editable_source_story` failed (length 0 rather than 1); copying the story for PDF compilation fixed it. Real Markdown sources now contain prose, actual result tables and figure links.
2. Resume checks bound to the data manifest but did not protect underlying annotation/image bytes. Regression `test_changed_annotation_bytes_are_rejected` failed before implementation. Each Case now stores SHA-256 for image, target and valid mask; loading verifies them, each observation stores those hashes, and resume independently verifies the actual files. The identical seeded split was prepared again; validation, selection and both held-out corpora were rerun with the repaired protocol. Model weights stayed frozen.

The reviewer returned no scientific arithmetic/leakage finding in its messages. Its final overall verdict was not returned because the reviewer process reached an account usage limit. The numerical audit and specific findings above were received before that interruption; do not describe this as an unconditional reviewer approval. Both returned Important issues were corrected and regression-tested by the main implementer.

Authoring fixes during visual QA: prevent caption clipping and image distortion; use two readable stable failures on slide 7; label native cost chart in milliseconds; embed a snapshot workbook with honestly rounded microsecond-resolution timings; validate a replacement PPT before copying it over the existing deliverable.
