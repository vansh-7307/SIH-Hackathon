# SignalScope
**Telling Real From Synthetic in the Age of Generative Media**

## Problem
Generative AI creates hyper-realistic synthetic media that challenges traditional detection methods. Current AI detectors often fail to generalize to **unseen generators**—they memorize specific generator artifacts rather than learning fundamental forensic signals.

## Solution
SignalScope is a research-oriented, production-ready AI media forensics system that explicitly targets out-of-distribution generalization. It triangulates authenticity using spatial visual features, frequency-domain artifacts, high-frequency residuals, and calibrated ensemble inference. 

## Architecture
```mermaid
graph TD
    Image[User Image] --> Spatial[Spatial Branch (ConvNeXt)]
    Image --> Freq[Frequency Branch (FFT)]
    Image --> Art[Artifact Branch (SRM Residuals)]
    
    Spatial --> Fusion[Attention Fusion]
    Freq --> Fusion
    Art --> Fusion
    
    Fusion --> Classifier[Classifier Ensemble]
    Classifier --> Calibrator[Temperature Scaling]
    
    Calibrator --> Verdict[Responsible Verdict]
    Calibrator --> Explain[Faithful Explanation]
    Image --> Robust[Degradation Robustness]
```

## Hackathon Modules Addressed
- **Core Task**: Generalizable Real vs AI Classification
- **Bonus A**: Faithful Explanation (Grad-CAM, Artifact Localization)
- **Bonus C**: Robustness (JPEG, Blur, Resize)
- **Bonus D**: Provenance & Metadata (EXIF parsing, processing software detection)
- **Bonus F**: Real-time deployment (FastAPI)

## Data Pipeline
SignalScope uses a custom split strategy ensuring that test images come from **unseen generators** to properly evaluate generalization, and implements strict leakage detection using SHA-256 hashing.

## Usage

### Local Inference
```bash
python model/predict.py --image path/to/image.jpg --demo
```

### API
```bash
docker-compose up -d
```
The FastAPI server will start on `http://localhost:8000`.

## Explanation and Robustness
The system produces localized explanations tied directly to the model's intermediate spatial representations. It also analyzes stability under JPEG compression and blurring to confirm robustness. Output is calibrated to provide a realistic probability estimate, preventing overconfident false accusations.

## Ethical Scope
SignalScope provides a likelihood assessment, not proof of fraud. It should not be used in isolation for adjudicating real-world events or making accusations against individuals.