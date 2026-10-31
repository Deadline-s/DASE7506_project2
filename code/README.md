# Installation and execution

Use Python 3.12 and run commands from the repository root. The root README contains exact installation, evaluation, reconstruction and ablation commands.

## Official files

common.py, predict.py, score_predictions.py, check_data.py, data/ and PACKAGE_MANIFEST.json are preserved from the upstream assignment. submission.py implements the required prediction interface. requirements.txt pins the inference environment.

## Released implementation

- submission.py and ranker_inference.py: candidate scoring and fixed ensemble inference.
- set_attention.py: train-only candidate resampling, neural training and NumPy attention inference.
- reasoning_features.py, verification_features.py, number_features.py and comparison_features.py: fixed numerical, textual and relative evidence.
- text_ranker.py: training and portable inference for lexical evidence.
- train.py, retrain.py, verified_training.py and training_data.py: reproduce the frozen recipe using supplied train references.
- kernel_training.py: reconstruct the RBF component of the initial comparison model; the released ensemble omits that branch.
- evaluate.py: fresh prediction and official scoring with four threads.
- verify_candidate.py: parameter comparison and inference invariants.
- analyze_attention_release.py: fixed-weight validation ablations.
- training_config.json and artifacts/model.pkl: matching training recipe and released checkpoint.

Only NumPy and model.pkl are needed for inference. Install requirements-training.txt for reconstruction. All active parameter arrays and normalization statistics are in model.pkl; no training corpus is loaded by the prediction method.

PACKAGE_MANIFEST.json is the original upstream manifest; the root SUBMISSION_MANIFEST.json records this submission. Removed search scripts and unused model-family branches are not required by the released recipe. Historical experiments remain documented in results/experiments.json.
