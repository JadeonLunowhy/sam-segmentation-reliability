# Environment and installation record

All installed Python packages belong to project/.venv; system Python and GPU drivers are unchanged.

Base interpreter observed: Python 3.13.4 at B:/Python/python.exe.
GPU observed: NVIDIA RTX 5060 Laptop GPU, 8 GB, driver 573.24 advertising CUDA 12.8. Real CUDA tensor multiplication and SAM inference passed. SAM peak allocated memory is approximately 2.9 GB; all parameters were verified frozen.

Commands used:

1. `python -m venv .venv`; ensurepip required a second run with TEMP/TMP set to project/tmp because of Windows temporary-directory access restrictions.
2. Local pip install: numpy, pillow, scipy, matplotlib, scikit-learn, reportlab, imageio-ffmpeg.
3. Local pyarrow install for the publicly mirrored Pet test images.
4. Official PyTorch 2.8.0+cu128 (3,461,390,892-byte wheel, SHA-256 9e20646802b7fc295c1f8b45fefcfc9fb2e4ec9cbe8593443cd2b9cc307c8405) and torchvision 0.23.0+cu128 installed in .venv, with transitive dependencies.
5. Official SAM inference source cloned into third_party/segment-anything at revision dca509fe793f601edb92606367a655c15ac00fdf.

Model: official sam_vit_b_01ec64.pth, 375042383 bytes; observed SHA-256 ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912.

Bundled artifact runtime located under C:/Users/31479/.cache/codex-runtimes/codex-primary-runtime. It is read without modification and provides JavaScript Artifact Tool 2.8.59 for editable presentations.

Exact installed versions are in requirements-lock.txt; observed runtime and device information is in results/environment.json. Pip logs and large transfer intermediates are in project/tmp.

Data-source ruling: official Oxford Pet image archive transport stalled; use the public timm/oxford-iiit-pet test Parquet mirror while retaining official test-list and trimap annotations. This changes transport, not the intended test corpus or its evaluation labels. Save the mirror URL/hash and selected filenames.
