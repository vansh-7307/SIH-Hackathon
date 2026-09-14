# SignalScope: One-Page Model Report

## Task
The primary task is to distinguish authentic images from AI-generated media while generalizing to unseen generators. The system must provide calibrated confidence and faithful visual explanations for its decisions.

## Data & Split
The dataset requires splitting such that the test set evaluates zero-shot transfer (generators present in the test set are explicitly excluded from the training and validation sets). We enforce a strict deduplication using SHA-256 to prevent leakage across splits.

## Model / Approach
We employ a multi-branch architecture:
1. **Spatial Branch**: Uses a ConvNeXt backbone for high-level semantics.
2. **Frequency Branch**: Computes 2D FFT log-magnitude spectrums to detect synthetic checkerboard patterns.
3. **Artifact Branch**: Applies fixed High-Pass SRM filters to isolate local texture inconsistencies.

These branches are combined via an **Attention Fusion** module and classified by an ensemble head. We calibrate the final probabilities using **Temperature Scaling** to prevent overconfidence.

## Metric & Result
*(Note: These are placeholders to be filled post-evaluation in the hackathon).*
- Overall ROC-AUC: Pending evaluation
- Unseen Generator AUC: Pending evaluation 
- Macro-F1: Pending evaluation
- FPR @ 95% TPR: Pending evaluation

## Baseline
A single-branch spatial-only ResNet/EfficientNet classifier serves as the baseline to demonstrate the performance lift of the multi-branch frequency fusion method.

## Limitations
- **Resolution**: Highly compressed or drastically downscaled images may destroy forensic frequency signals.
- **Novel Architectures**: Radically new generative models may not exhibit the frequency artifacts the model has learned to detect.
- The system is probabilistic and intended as an investigative aid, not absolute proof.
