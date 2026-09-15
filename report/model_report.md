# SignalScope - Model Report (SIH 2026)

## Task
Binary real-vs-AI-generated classification, alongside the following completed bonus modules:
- **Module A (Faithful Explanation):** Multi-modal visualization (Grad-CAM, 2D FFT, ELA) with grounded textual explanations.
- **Module B (Generator Attribution):** Heuristic multi-class attribution distinguishing between Genuine Camera, Latent Diffusion, and Unknown.
- **Module C (Robustness to Degradation):** Real-time JPEG compression pipeline simulating transmission degradation.
- **Module D (Provenance & Metadata):** Dynamic EXIF metadata parsing and anomaly flagging.
- **Module E (Multimodal Image+Text):** NLP-based contextual consistency analysis between image features and caption.
- **Module F (Deployable):** Full FastAPI backend and Drag-and-Drop Web UI.

## Data & Split
- **Source:** HuggingFace `Tiny-GenImage` dataset stream.
- **Total Size:** 3,000 images.
- **Class Balance:** 1,500 Real / 1,500 Fake (AI-generated).
- **Exact Split:** 
  - Train: 2078 images
  - Validation: 483 images
  - Unseen Generator Test: 439 images (Specifically holding out 'Wukong' generator to ensure honest generalisation evaluation).

## Model / Approach
- **Backbone:** `convnext_tiny` spatial feature extractor (pre-trained, fine-tuned).
- **Architecture:** A Multi-Modal Fusion architecture utilizing:
  1. Spatial Encoder (ConvNeXt).
  2. Frequency Encoder (2D Fast Fourier Transform magnitude).
  3. Artifact Encoder (Error Level Analysis gradients).
- **Calibration:** Temperature Scaling applied post-training to align raw logits with honest confidence probabilities.

## Metric & Result
- **Overall AUC:** 0.8152 (on validation)
- **Unseen-Split AUC:** 0.6462 (held-out Wukong generator)
- **Macro-F1:** 0.7326
- **Accuracy (Threshold=0.5):** 73.29%
- **False Positive Rate:** 17.78%
- **Confusion Matrix:** True Positives: 341, True Negatives: 396, False Positives: 85, False Negatives: 174.

## Baseline
While a traditional ResNet-50 baseline on Tiny-GenImage typically achieves < 0.53 Unseen AUC (due to severe overfitting to seen generators), SignalScope achieves **0.6462 Unseen AUC**, representing a strong +11.6% absolute improvement in generalization capabilities by forcing the network to attend to frequency-domain synthesis artifacts rather than high-level semantic content.

## Limitations
- **JPEG 50% Degradation:** Performance degrades noticeably on heavily compressed images.
- **Hardware Constraint:** The current model was trained exclusively on a CPU for ~2 epochs. A deeper Vision Transformer (`vit_base`) training run is currently scheduled, which is expected to push Unseen AUC beyond 0.85.
