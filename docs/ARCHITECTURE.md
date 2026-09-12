# Architecture & Engineering Decisions

This document records the significant design decisions in the Skin Lesion Classifier,
including alternatives considered and their trade-offs. It complements the README, which
covers *how* to run the project; this document covers *why* it is built this way.

## System overview

```
User ──> Browser ──> Flask app (app.py) ──> Preprocessing ──> EfficientNet-B0 (torch)
                          │                                          │
                          │                                          ▼
                          │                             Softmax probabilities
                          │                                          │
                          ▼                                          ▼
                    Jinja templates                        SQLite history log
                    (index / batch / history)              (history.py)
```

There is no separate backend service, no message queue, and no external model server.
The Flask process owns request routing, image preprocessing, model inference, and
persistence. This is deliberate — see decisions below.

---

## Decision 1: Single-process Flask app

**Decision.** All request handling, preprocessing, and inference run inside one Flask
process (`app.py`). No worker queue or separate model service.

**Why.**
- The workload is a portfolio / educational demo. Inference is CPU-bound and takes on
  the order of hundreds of milliseconds per image on a laptop — well within Flask's
  synchronous request model.
- Deploying a single process is simpler for reviewers to reproduce.
- Keeping preprocessing and inference in the same process avoids serializing tensors
  over IPC or HTTP.

**Alternatives considered.**
- FastAPI + Celery + Redis worker for async inference.
- TorchServe / a dedicated model server behind a lightweight web layer.

**Trade-off.** Under real production load, the model call blocks the request thread and
would need Gunicorn workers or an async architecture. For this demo, the simplicity is
worth more than the throughput.

---

## Decision 2: Flask over FastAPI

**Decision.** Use Flask 3 for the web layer.

**Why.**
- Server-rendered Jinja templates are already the primary UX — a single-page React
  frontend would be significant scope creep for the value it adds here.
- Flask is minimal, well-known, and stable; the whole app is small enough that FastAPI's
  extra ergonomics (type-driven validation, async, auto docs) do not pay off.

**Alternatives considered.**
- FastAPI: better for a pure JSON API, but the UI is the primary interface.
- Django: too heavy for a two-model-endpoint project.

**Trade-off.** No automatic OpenAPI docs. The API surface (four endpoints) is small
enough that the README covers it clearly.

---

## Decision 3: EfficientNet-B0 as the backbone

**Decision.** Use `torchvision.models.efficientnet_b0` with a replaced classifier head
(`Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→7)`), loaded from a fine-tuned
checkpoint at runtime.

**Why.**
- EfficientNet-B0 has a strong accuracy-to-parameter ratio for 224×224 image
  classification on modest hardware.
- The custom head lets a downstream fine-tuning workflow keep the ImageNet features and
  learn a small dermoscopy-specific head.
- Torchvision is already a hard dependency for the transforms — no additional framework.

**Alternatives considered.**
- ResNet50: more parameters, marginal accuracy gains on this size of dataset.
- MobileNetV3 / smaller: faster but weaker features.
- Timm models: broader zoo but a heavier dependency for one architecture.

**Trade-off.** Model file is roughly 20 MB; still small enough to ship or download
alongside the code, but too large to commit — see Decision 4.

---

## Decision 4: Model weights out of the repository, loaded via `MODEL_PATH`

**Decision.** The trained `.pth` checkpoint is not committed to git. `app.py` reads its
location from the `MODEL_PATH` environment variable, defaulting to
`./model/skin_disease_classification_model.pth`.

**Why.**
- Trained weights are large binary artifacts; git is not the right store.
- Different reviewers may have different checkpoints; the env-var indirection keeps the
  code the same across setups.
- If the checkpoint is missing at startup, the app logs a warning and boots with
  `model = None`. Requests receive a clear "Model is not loaded" message rather than a
  500. The UI, tests, and health check all still work.

**Trade-off.** A first-time reviewer sees the UI without predictions unless they
provide a checkpoint. The README documents this explicitly.

---

## Decision 5: Image preprocessing pipeline

**Decision.** Resize to 224×224 → ToTensor → ImageNet mean/std normalization
(`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).

**Why.**
- Matches the input distribution EfficientNet-B0's ImageNet-pretrained backbone expects.
- The fine-tuned head was trained with this transform, so inference must use the same
  one to avoid silent accuracy loss.

**Trade-off.** No test-time augmentation, no center-crop-after-resize (which would
better preserve aspect ratio on non-square inputs). Kept simple to match the training
pipeline exactly.

---

## Decision 6: SQLite for prediction history

**Decision.** Persist every prediction (web, API, batch) to a local SQLite database
(`predictions.db`) via the standard-library `sqlite3` module.

**Why.**
- Zero-dependency, single-file store; matches the deployment simplicity of the Flask
  process.
- Reads and writes are fast enough at demo scale (single-digit connections).
- Easy to inspect from any client, and easy for tests to point at a temp path.

**Alternatives considered.**
- Postgres: overkill for a demo.
- JSON log file: harder to query for the risk-count summary.

**Trade-off.** SQLite does not scale to many concurrent writers. If this ever became a
real service, we would swap to Postgres. The `history.py` module is small enough that
this is a one-file change.

---

## Decision 7: `uv` for Python dependency and environment management

**Decision.** All dependencies live in `pyproject.toml`; the environment is provisioned
with `uv sync`; the app is run with `uv run …`. `requirements.txt` has been removed. The
project is declared as non-packaged (`[tool.uv] package = false`) because we don't
publish it — we just run it.

**Why.**
- Single command (`uv sync`) creates the venv, resolves the graph, and installs
  everything; `uv.lock` guarantees reproducibility.
- `uv` is dramatically faster than pip for a torch-sized graph.
- One canonical dependency source (`pyproject.toml`) instead of duplicated
  `requirements.txt` + setup files.

**Alternatives considered.**
- pip + `requirements.txt`: what the project used before. Reliable but slow and requires
  a separate lockfile tool for reproducibility.
- Poetry / Hatch / PDM: workable but slower and heavier than `uv`.

**Trade-off.** `uv` is a newer tool; anyone unfamiliar with it needs a one-line install
step. The README covers this.

---

## Decision 8: Class list and risk-level mapping baked into the app

**Decision.** The seven HAM10000-style classes (`akiec`, `bcc`, `bkl`, `df`, `mel`,
`nv`, `vasc`) and their three-bucket risk mapping (high / medium / low) are constants in
`app.py`.

**Why.**
- These are model-defined: they must match the checkpoint's output layer order.
- Keeping them next to `run_inference` makes the coupling obvious.

**Trade-off.** A different checkpoint (different class set) requires a code edit rather
than a config file. Acceptable at this scale — the alternative (a class-map JSON file)
would add a moving part without adding value.

---

## Decision 9: Medical-safety framing

**Decision.** Every user-facing surface (each template, the README, this doc) states
explicitly that the tool is an educational demo, uses the phrase "model prediction"
rather than "diagnosis", and never claims that a prediction is a medical finding.

**Why.**
- The application is a machine-learning demonstration, not a medical device. Framing
  its outputs as diagnoses would be actively harmful.
- The disclaimer must be visible before the user acts on a result, not buried in the
  footer — so it is a banner near the top of each page.

**Trade-off.** The UI looks slightly busier. That is the correct trade-off.

---

## Decision 10: Test strategy — real inference, not mocks

**Decision.** The test suite (`tests/`) builds a randomly-initialized EfficientNet-B0
with the same head as production, saves it to a temp path, and points `MODEL_PATH` at
it. Tests then hit the real Flask routes through `app.test_client()` and exercise the
actual `torch` model.

**Why.**
- Catches drift between preprocessing, inference, and result shape that mocks would
  hide.
- Runs on CPU in a few seconds — no GPU or trained checkpoint required.
- Tests represent what breaks the app for a real user, not what breaks a stub.

**Trade-off.** Tests need `torch` installed (heavy) rather than just Flask (light).
Given that `torch` is a production dependency anyway, the extra install cost is zero.
