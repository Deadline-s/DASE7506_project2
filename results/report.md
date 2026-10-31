# Candidate-Set Verification
DASE7506 Project 2 | 31 October 2026
## Research question
Can interactions among supplied candidate solutions and train-only balancing of correct-candidate counts improve answer selection under a 16 MiB asset budget and a 30-second CPU evaluation limit? We hypothesize that a verifier should compare evidence across candidates rather than rely mainly on each candidate's plausibility or the majority answer.
## Task and evaluation protocol
For each question, select exactly one of four supplied candidates. The unchanged official scorer awards credit when the selected final numeric answer is equivalent to the reference; it does not verify every reasoning step. The supplied banks contain 4,755 training, 798 validation and 467 test questions. Fixed embeddings, candidate text and question text are available; inference generates no new solutions.
Learned parameters use only supplied train references. Validation is used for development, mixture selection and ablations. We keep the supplied data, numeric parser, prediction driver and scorer unchanged, and report the official complete-test accuracy.
## Main result
| Method | Validation correct | Test correct |
| --- | --- | --- |
| Initial numerical/neural/lexical v9 | 746/798 | 387/467 (82.869379%) |
| Released candidate-set ensemble | 756/798 | 392/467 (83.940043%) |
| Later validation-selected ensemble | 757/798 | 389/467 (83.297645%) |
| Later unit-aware ensemble | 757/798 | 389/467 (83.297645%) |
The released ensemble improves the initial verifier by five test questions, or 1.071 percentage points. Later one-question validation improvements do not transfer to the test bank. The evidence supports evaluating both accuracy and generalization rather than treating small development gains as conclusive.
## Selection limitation
The released configuration was originally frozen before its first formal test. The final package returns to it after multiple aggregate test scores were known, so the release choice is post hoc. This is not a blind selection procedure. No per-question test errors were inspected and no test references were used to fit parameters. The assignment prohibits test-based selection; this history is disclosed for the instructor's assessment rather than represented as unqualified compliance.
---PAGE---
# Method and Training
## Baseline evidence
The initial verifier combines supplied embeddings with answer-count and text-style features, arithmetic consistency checks, per-candidate neural rankers, lexical comparisons and an RBF branch. Equivalent numeric answers are pooled using the official parser. Such evidence can distinguish clear mistakes, but consistent arithmetic can still implement the wrong interpretation of a story problem.
## Candidate-set attention
Each candidate is represented by its 1,024-dimensional fixed embedding, 27 compact candidate features and 32 verification features. A learned projection maps these inputs to width 96. One pre-normalized transformer layer uses four attention heads, a feed-forward width of 192, ReLU activation and residual connections. A final normalization and linear readout produce candidate logits; inference uses exported NumPy parameters.
The objective combines binary cross entropy with 0.4 times a correct-answer-mass loss. The mass loss is logsumexp over valid candidates minus logsumexp over correct candidates. Groups containing an incorrect candidate receive twice the weight of all-correct groups. AdamW uses learning rate 0.0003, weight decay 0.01, dropout 0.2, batch size 128 and gradient norm clipping at 1. Training lasts eight epochs.
## Train-only candidate balancing
Within each training fold, resample existing correct and incorrect candidates to create groups with one, two or three correct candidates. Recompute group-dependent features after resampling. No external text or new answers are generated. A plain branch uses original groups; a balanced branch adds these resampled groups. Each branch averages five models, each fitted on four of five stratified train folds. The fold seed is 941 and model seeds are 8341-8345.
## Fixed ensemble
| Branch | Weight |
| --- | --- |
| Plain ReLU candidate-set attention | 0.20 |
| Balanced ReLU candidate-set attention | 0.60 |
| Existing numerical/neural/lexical verifier without RBF | 0.20 |
Candidate probabilities are averaged within each equivalent-answer group, then normalized across unique answers before fusion. Removing the RBF branch frees asset space for candidate interaction networks. The existing branch retains its frozen configuration and seeds 7506-7511 as specified in training_config.json.
---PAGE---
# Experiments and Ablations
## Development comparisons
We compared the initial verifier with the candidate-set ensemble using the same supplied splits and official evaluation. Development explored input features, activation functions, loss weights, regularization, early checkpoints, parameter averages, full-training refits, initialization repeats and ensemble allocations. More complex OOF trees, pairwise contexts and arithmetic-program evidence were also investigated; most did not improve the validation incumbent. Historical configurations, metrics and study durations are retained in results/experiments.json.
The later validation-selected ensemble replaces the plain attention branch with a GELU attention variant and changes mixture weights. A further unit-aware variant extends arithmetic evidence. Both obtain 757/798 on validation but 389/467 on test. Neither is part of the frozen checkpoint. These comparisons illustrate development overfitting and uncertainty after repeated adaptive trials.
## Fixed-weight branch removal
For each ablation, remove one released branch and renormalize the remaining fixed weights. Do not retrain or optimize the surviving branches. These comparisons use validation only.
| Variant | Correct | Accuracy |
| --- | --- | --- |
| Full released ensemble | 756/798 | 94.74% |
| Remove plain | 753/798 | 94.36% |
| Remove balanced | 745/798 | 93.36% |
| Remove existing | 750/798 | 93.98% |
Removing balanced attention loses 11 validation questions, compared with three for plain attention and six for the existing branch. This suggests complementary evidence in the frozen mixture, with the balanced branch contributing most under this ablation. It does not isolate augmentation alone: comparing separate trained branches and renormalized mixtures also changes interactions and calibration.
## Rejected alternatives
A bounded question-to-program classifier considered 387 arithmetic programs, covering 2,381 of 4,755 training answers. It did not improve validation and was not adopted or test-scored. Hand-designed question grammars suggested improvements on a few inspected validation cases, but their train coverage was only one, zero and zero matched questions; the train-coverage gate disabled them. Neither mechanism is used by the released model.
---PAGE---
# Error Analysis and Resources
## Validation subgroups
| Number of correct candidates | Correct questions | Accuracy |
| --- | --- | --- |
| 1 | 83/117 | 70.94% |
| 2 | 117/123 | 95.12% |
| 3 | 177/179 | 98.88% |
| 4 | 379/379 | 100.00% |
The one-correct-candidate group is the most difficult: the correct answer must overcome several plausible distractors. The 379 all-correct groups are automatically solvable under this task's answer-selection metric and inflate aggregate validation accuracy. Performance on minority-correct groups is therefore especially informative.
Numerical consistency is necessary evidence but not sufficient reasoning verification. Typical inspected validation failures involve applying an average to the wrong collection of scores or confusing a total population with unmatched members. Multiple candidates can share the same incorrect interpretation. No individual test failures were inspected.
## Cost and reproducibility
The only inference asset is model.pkl: 15,137,411 bytes (14.436 MiB), below 16 MiB uncompressed. Learned statistics and all ensemble parameters are included. Inference uses NumPy on CPU, with no network calls, external model weights, new generation or answer lookup.
A fresh four-thread prediction process followed by independent official scoring takes 1.985 seconds for the complete test bank on the recorded arm64 macOS host. Timing includes startup, imports, loading, preprocessing and all ensemble members. The official Xeon Platinum 8457C reference machine has not been measured; local timing is not a reference-machine guarantee.
A fixed-recipe reconstruction after source cleanup takes 54.14 seconds. Its 461 arrays and all metadata exactly match the released checkpoint. Pickle byte hashes can differ because of shared-reference serialization. Both hashes and the reconstruction time are recorded in results/training.json. Historical study durations are recorded separately; overlapping runs are not summed as elapsed wall time.
## Limits of the evidence
Repeated validation reuse creates selection bias, and differences of one or a few questions are not established generalization gains. Ablations test the chosen mixture rather than every design decision independently. The final release choice used already-known aggregate test results, as disclosed on page 1. No further optimization is performed after freezing.
---PAGE---
# Reproduction and Attribution
## Running the frozen checkpoint
Use Python 3.12 and install code/requirements.txt. From the project root, run python code/check_data.py and python code/evaluate.py. The evaluator invokes the official predictor and scorer with four-thread limits, then writes official score JSONs and resource measurements to results/.
The expected scores are test 392/467 (83.940043%) and validation 756/798 (94.736842%). The root README provides copyable commands, including isolated output directories for reproducing the initial comparison.
## Training and validation
Install code/requirements-training.txt and run python code/train.py --output /tmp/mp2-reproduced.pkl. The default recipe is code/training_config.json; an explicit output keeps the release intact. Compare it using code/verify_candidate.py with --model code/artifacts/model.pkl, --reproduced /tmp/mp2-reproduced.pkl, --groups 798 and an output JSON path.
All 798 validation groups pass finite-output, batch-partition, candidate-permutation, question-order, arbitrary-ID and empty-input checks. Run code/analyze_attention_release.py with the released model and an output JSON path to reproduce fixed-weight ablations. Official files are checked against upstream commit bb801f4f2e7629d270df765ab12c333b2dc616bb.
## Version and assets
Release identifier: mp2-final-20261031. The student creates and publishes the matching version tag when submitting. Source and artifact checksums are recorded in SUBMISSION_MANIFEST.json. Released checkpoint SHA-256:
22888b2c568a1dfcb06443db27dd7403 79363d681a8680c18a07aea70796bbc6
The final code retains inference, fixed-recipe training, the initial comparison and validation tools. Unused search entry points and experimental model families are omitted. results/experiments.json preserves the historical evidence; old paths inside that record describe earlier experiments.
## References and AI disclosure
Assignment: xudongwu-0/DASE7506-MP2, commit bb801f4f2e7629d270df765ab12c333b2dc616bb. Supplied GSM8K questions and Qwen3 candidate representations retain their license and attribution files.
Conceptual reference: Wenqi-Ge/DASE7506-MP2, commit 54abbb793ddb6bd2142aa8b5cf56c5a0b0681b88. Candidate-count balancing and candidate attention motivated independent implementations with our own features, folds, seeds and recipes. The peer reports 402/467; we did not independently reproduce it, and no peer weights, prediction files or implementation files are included.
OpenAI Codex assisted design, coding, experiments, verification, cleanup and documentation. All model fitting uses supplied train references. No external dataset, additional pretrained checkpoint or test-answer lookup is used.
