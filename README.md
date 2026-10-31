# DASE7506 Project 2: Candidate-Set Verification

Frozen release: mp2-final-20261031. Official complete-test selection accuracy: **83.940043% (392/467)**. Validation: **94.736842% (756/798)**.

The verifier selects one of four supplied candidates using numerical evidence, candidate-set attention and a fixed ensemble. The report presents the research question, baseline comparison, validation ablations, failure analysis, resource measurements and limitations.

## Repository layout

- code/: unchanged official data and evaluation tools; implemented inference, training and verification modules.
- results/: official score JSONs, fixed-recipe training and validation evidence, historical experiment records and the report source.
- output/pdf/report.pdf: five-page project report.
- SUBMISSION.md: publication commands and submission-form instructions for the student.
- SUBMISSION_MANIFEST.json: matching release identifier, asset hash and file checksums.

## Installation and evaluation

Use Python 3.12. From the repository root:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r code/requirements.txt
    python code/check_data.py
    python code/evaluate.py

The evaluator runs the supplied prediction driver and scorer in fresh processes with four-thread limits. It regenerates predictions, score JSONs and resource measurements in results/. Saved predictions are reproducible outputs and are not required inference assets.

## Reproduce the training recipe

    python -m pip install -r code/requirements-training.txt
    python code/train.py --output /tmp/mp2-reproduced.pkl
    python code/verify_candidate.py --model code/artifacts/model.pkl --reproduced /tmp/mp2-reproduced.pkl --groups 798 --output /tmp/mp2-verification.json

The default configuration is code/training_config.json. It records architecture, losses, epochs, mixture weights and seeds. Supplying --output protects the released checkpoint; omitting it replaces the default artifact. The retained code reconstructs the released model and the initial v9 comparison. It does not include unused experimental model families or search entry points.

To reproduce the initial comparison:

    python code/train.py --config results/baseline-v9-training-config.json --output /tmp/mp2-baseline/model.pkl
    mv /tmp/mp2-baseline/model.training.json /tmp/mp2-baseline-training.json
    python code/evaluate.py --checkpoint /tmp/mp2-baseline --output-dir /tmp/mp2-baseline-results

The move keeps the training record outside the inference-asset folder.

Validation ablations:

    python code/analyze_attention_release.py --model code/artifacts/model.pkl --output /tmp/mp2-ablation.json

## Result and resources

| Method | Validation correct | Test correct | Test accuracy |
| --- | --- | --- | --- |
| Initial v9 verifier | 746/798 | 387/467 | 82.869379% |
| Candidate-set ensemble, released | 756/798 | 392/467 | 83.940043% |
| Later validation-selected ensemble | 757/798 | 389/467 | 83.297645% |
| Later unit-aware ensemble | 757/798 | 389/467 | 83.297645% |

- One inference asset: code/artifacts/model.pkl; 15,137,411 bytes (14.436 MiB), below the 16 MiB limit.
- Four-thread fresh complete-test prediction plus scoring: 1.985 seconds on the recorded arm64 macOS host. The official Xeon reference machine has not been measured.
- SHA-256: 22888b2c568a1dfcb06443db27dd740379363d681a8680c18a07aea70796bbc6.
- After source cleanup, fixed-recipe reconstruction matches 461 arrays and all metadata; all 798 validation groups pass inference invariance checks.
- The original data, parser, prediction driver, scorer and guide match the upstream repository. See results/release-checks.json.

## Experimental provenance and disclosure

The first attention configuration was frozen before its first formal test. The final release returns to that checkpoint after historical aggregate test scores were known. This is a post-hoc release choice, not blind test selection; it is disclosed in the report. No individual test errors are inspected and no test references fit parameters. Repeated validation use also introduces selection bias.

Wenqi-Ge/DASE7506-MP2, commit 54abbb793ddb6bd2142aa8b5cf56c5a0b0681b88, inspired candidate-count balancing and attention. Its reported score was not independently reproduced. We use independently written modules and our own features, recipes and seeds; no peer weights, predictions or implementation files are incorporated. OpenAI Codex assisted design, implementation, experiments, verification and documentation. No external training corpus or additional pretrained weights are used. Supplied licenses and attribution are retained.

results/experiments.json contains historical searches, durations and candidate evaluations; its old paths and hashes identify historical experiments, not the current file layout. The root manifest identifies the submitted source and checkpoint. Redundant scripts and temporary records are excluded from the final package.
