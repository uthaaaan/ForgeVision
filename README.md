# ForgeVision - AI-Powered Industrial Defect Detection

ForgeVision is a computer vision project that detects industrial product defects using normal-only training data. It implements and compares three anomaly detection approaches: Convolutional Autoencoder, PaDiM, and PatchCore.

The project uses the MVTec Anomaly Detection (MVTec AD) dataset, specifically its bottle category.

The objective is to identify defective products without requiring defective images during model training and to generate pixel-level anomaly maps that highlight suspicious regions.

## Project Overview

Industrial quality inspection requires identifying defects such as scratches, cracks, and surface irregularities. Traditional manual inspection can be time-consuming and inconsistent.

ForgeVision explores how deep learning and anomaly detection can automate this process.

Unlike conventional supervised classification, the models are trained using only normal product images. They learn normal visual patterns and identify deviations during inference.

### Key Features

- Normal-only training for anomaly detection.
- Three different anomaly detection algorithms.
- Image-level normal and defective classification.
- Pixel-level anomaly localization.
- Anomaly heatmap generation.
- Model performance comparison.
- Calibrated decision thresholds.
- Interactive Streamlit application.
- GPU-accelerated training and inference.

## Models Implemented

ForgeVision implements three anomaly detection approaches.

### 1. Convolutional Autoencoder

The Autoencoder learns to reconstruct normal bottle images.

It consists of an encoder and a decoder. The encoder compresses the input image into a lower-dimensional representation, while the decoder attempts to reconstruct the original image.

During inference, the original image is compared with its reconstruction. Regions with larger reconstruction errors receive higher anomaly scores.

**Main concepts:**
- Convolutional Neural Networks
- Encoder-decoder architecture
- Bottleneck representations
- Reconstruction error
- Mean Squared Error loss
- Backpropagation and optimization

### 2. PaDiM - Patch Distribution Modeling

PaDiM uses a pretrained ResNet18 feature extractor to obtain visual representations from multiple network layers.

It models the distribution of normal feature vectors at individual spatial locations using multivariate Gaussian distributions.

During inference, Mahalanobis distance measures how unusual each feature vector is compared with the learned normal distribution.

**Main concepts:**
- Transfer learning
- ResNet18 feature extraction
- Multi-level feature representations
- Multivariate Gaussian distributions
- Mean and covariance estimation
- Mahalanobis distance
- Covariance regularization

### 3. PatchCore

PatchCore extracts local feature representations using a pretrained ResNet18.

It builds a memory bank containing representative normal feature patches. During inference, new feature patches are compared with the memory bank using nearest-neighbor distance.

Patches that are significantly different from the stored normal representations receive higher anomaly scores.

A coreset selection strategy reduces the memory bank size while attempting to preserve representative normal patterns.

**Main concepts:**
- Deep feature extraction
- Patch-based anomaly detection
- Nearest-neighbor search
- Euclidean distance
- Memory banks
- Greedy farthest-point sampling
- Coreset selection

## Dataset

**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**Category:** Bottle  
**Original image resolution:** 900 x 900  
**Processed image resolution:** 256 x 256

The dataset contains normal training images, normal test images, defective test images, and ground-truth segmentation masks.

### Dataset Split

The 209 available normal training images were divided into two groups:

| Split | Images | Purpose |
|---|---:|---|
| Training | 167 | Model training |
| Calibration | 42 | Threshold calibration |

The independent test set contains:

| Test category | Images |
|---|---:|
| Normal | 20 |
| Defective | 63 |
| Total | 83 |

All three models were trained using the same 167 normal training images.

The 42 calibration images were used only to determine model-specific decision thresholds. They were not used for model fitting.

The test set remained separate from model training and threshold calibration.

### Dataset License

The MVTec AD dataset is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0).

Dataset users must comply with the license terms, including attribution, non-commercial use, and applicable share-alike requirements.

Dataset source: https://www.mvtec.com/company/research/datasets/mvtec-ad

## Project Architecture

The overall workflow is:

```text
                 MVTec AD Dataset
                        |
                        v
                 Dataset Preparation
                        |
                        v
                 Normal Image Split
                   /          \
                  /            \
                 v              v
          Training Split    Calibration Split
             167 Images       42 Images
                 |
                 v
          Model Training
                 |
       +---------+---------+
       |         |         |
       v         v         v
   Autoencoder  PaDiM   PatchCore
       |         |         |
       +---------+---------+
                 |
                 v
          Saved Checkpoints
                 |
                 v
        Threshold Calibration
                 |
                 v
         Independent Testing
                 |
                 v
        Performance Evaluation
                 |
                 v
          Inference Engine
                 |
                 v
          Streamlit Application
```

## Experimental Configuration

### Convolutional Autoencoder

| Parameter | Value |
|---|---|
| Architecture | Convolutional Autoencoder |
| Input size | 256 x 256 |
| Training samples | 167 |
| Epochs | 50 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error |
| Random seed | 42 |
| Final training loss | 0.00039899 |

### PaDiM

| Parameter | Value |
|---|---|
| Feature extractor | Pretrained ResNet18 |
| Feature layers | layer1, layer2, layer3 |
| Combined feature dimensions | 448 |
| Selected feature dimensions | 100 |
| Feature-map resolution | 64 x 64 |
| Training samples | 167 |
| Covariance regularization | 0.01 |
| Random seed | 42 |

### PatchCore

| Parameter | Value |
|---|---|
| Feature extractor | Pretrained ResNet18 |
| Feature layers | layer2, layer3 |
| Feature dimensions | 384 |
| Training samples | 167 |
| Extracted feature patches | 171,008 |
| Candidate subset | 8,192 |
| Memory bank size | 1,024 |
| Selection method | Greedy farthest-point sampling |

## Threshold Calibration

Anomaly detection models generate numerical anomaly scores. A decision threshold is required to classify an image as normal or defective.

ForgeVision uses the 95th percentile of the anomaly scores from the 42 normal calibration images.

The threshold is calculated independently for each model:

| Model | Calibrated Threshold |
|---|---:|
| Convolutional Autoencoder | 0.16890289 |
| PaDiM | 16.54107056 |
| PatchCore | 0.70110825 |

The image-level anomaly score is calculated using the maximum value of the anomaly map.

The classification rule is:

```text
If anomaly score > threshold:
    Prediction = Defective
Else:
    Prediction = Normal
```

The thresholds are empirical estimates based on a relatively small calibration set. They do not guarantee a particular false-positive rate on future production data.

## Evaluation Metrics

ForgeVision evaluates both image-level detection and pixel-level anomaly localization.

### Image AUROC

Image AUROC measures how effectively a model ranks defective images above normal images.

Each image receives one anomaly score.

### Pixel AUROC

Pixel AUROC measures how effectively a model assigns higher anomaly scores to defective pixels than normal pixels.

The ground-truth segmentation masks are used for pixel-level evaluation.

### Inference Time

Inference time measures the average time required to generate predictions under the evaluation setup.

## Final Experimental Results

All three models were evaluated on the same independent test set containing 83 images.

| Model | Image AUROC | Pixel AUROC | Average Inference (ms) |
|---|---:|---:|---:|
| Convolutional Autoencoder | 0.8905 | 0.7224 | 31.34 |
| PaDiM | 0.9929 | 0.9819 | 32.72 |
| PatchCore | 1.0000 | 0.9813 | 8.28 |

### Results Interpretation

**Convolutional Autoencoder**

The Autoencoder achieved an image AUROC of 0.8905 and a pixel AUROC of 0.7224. It provides a reconstruction-based baseline but performs less effectively than the feature-based methods in this experiment, particularly for pixel-level localization.

**PaDiM**

PaDiM achieved an image AUROC of 0.9929 and the highest pixel AUROC of 0.9819. Its statistical modeling of pretrained visual features provides strong image-level detection and pixel-level localization.

**PatchCore**

PatchCore achieved an image AUROC of 1.0000 and a pixel AUROC of 0.9813. It also achieved the fastest measured inference time of 8.28 ms.

### Overall Comparison

- Best image-level AUROC: PatchCore.
- Best pixel-level AUROC: PaDiM.
- Fastest measured inference: PatchCore.

PatchCore is particularly promising when image-level detection performance and inference speed are important. PaDiM provides slightly better pixel-level localization in this experiment.

The perfect image AUROC achieved by PatchCore applies only to this particular test set. It should not be interpreted as a guarantee of perfect performance on unseen real-world production data.

## Visual Results

The project generates several visual reports to compare model performance and understand anomaly localization.

### Model Comparison

`reports/model_comparison.png`

Provides a visual comparison of the three implemented models.

### Multi-Sample Anomaly Comparison

`reports/multi_sample_anomaly_comparison.png`

Compares five samples across five columns:

1. Original image.
2. Ground-truth defect mask.
3. Autoencoder anomaly map.
4. PaDiM anomaly map.
5. PatchCore anomaly map.

### Anomaly Comparison

`reports/anomaly_comparison.png`

Provides anomaly visualization results.

### Autoencoder Reconstruction

`reports/autoencoder_heatmap_comparison.png`

Compares original images, reconstructions, and reconstruction-based anomaly visualizations.

### Training History

`reports/autoencoder_loss_curve.png`

Shows the Autoencoder training loss progression.

### Dataset Preview

`reports/bottle_dataset_preview.png`

Provides examples from the processed bottle dataset.

## Streamlit Application

ForgeVision includes an interactive application built with Streamlit.

Run the application using:

```powershell
streamlit run app.py
```

The application allows users to:

- Upload a bottle image.
- Run all three anomaly detection models.
- View normal or defective predictions.
- Compare anomaly scores with calibrated thresholds.
- Inspect model inference times.
- View individual anomaly heatmaps.
- Examine heatmap overlays with the original image.

The application uses the shared inference engine to load trained checkpoints and generate predictions.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and pretrained models |
| NumPy | Numerical computations |
| Pandas | Dataset and results management |
| Matplotlib | Visualization |
| Pillow | Image processing |
| OpenCV | Image processing utilities |
| Scikit-learn | Evaluation metrics |
| SciPy | Scientific and numerical operations |
| Streamlit | Interactive application |
| Hugging Face Hub | Dataset access |
| CUDA | GPU acceleration |

## Hardware and Environment

The project was developed and evaluated using:

| Component | Configuration |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.14.1+cu126 |
| Torchvision | 0.29.1+cu126 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU Memory | 6 GB |
| CUDA | Available |

Training and inference performance may vary depending on hardware, software versions, and execution conditions.

## Project Structure

```text
ForgeVision/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- configs/
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- models/
|   |-- autoencoder_best.pth
|   |-- padim_parameters.pth
|   |-- patchcore_memory_bank.pth
|   |-- calibration_thresholds.json
|
|-- notebooks/
|
|-- reports/
|   |-- anomaly_comparison.png
|   |-- autoencoder_heatmap_comparison.png
|   |-- autoencoder_loss_curve.png
|   |-- autoencoder_training_history.csv
|   |-- baseline_results.csv
|   |-- bottle_dataset_preview.png
|   |-- calibration_scores.csv
|   |-- model_comparison.png
|   |-- multi_sample_anomaly_comparison.png
|   |-- reconstruction/
|
|-- src/
|   |
|   |-- data/
|   |   |-- load_data.py
|   |   |-- prepare_dataset.py
|   |
|   |-- models/
|   |   |-- autoencoder.py
|   |   |-- padim.py
|   |   |-- patchcore.py
|   |
|   |-- training/
|   |   |-- train_autoencoder.py
|   |   |-- train_padim.py
|   |   |-- train_patchcore.py
|   |
|   |-- evaluation/
|   |   |-- create_calibration_split.py
|   |   |-- calibrate_thresholds.py
|   |   |-- evaluate_autoencoder.py
|   |   |-- evaluate_padim.py
|   |   |-- evaluate_patchcore.py
|   |   |-- save_baseline_results.py
|   |   |-- save_padim_results.py
|   |   |-- save_patchcore_results.py
|   |
|   |-- inference/
|   |   |-- inference_engine.py
|   |   |-- test_inference.py
|   |
|   |-- visualization/
|       |-- compare_anomaly_maps.py
|       |-- compare_autoencoder_heatmap.py
|       |-- compare_models.py
|       |-- inspect_dataset.py
|       |-- inspect_reconstruction.py
|       |-- plot_training_history.py
|
|-- tests/
```

The dataset and large model checkpoint files are excluded from the Git repository to keep the repository lightweight. The calibration threshold JSON file is included.

## Installation and Usage

### 1. Clone the Repository

```powershell
git clone https://github.com/uthaaaan/ForgeVision.git
cd ForgeVision
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch and Torchvision build compatible with your NVIDIA GPU and CUDA environment. Follow the official PyTorch installation instructions if your environment requires a different build.

### 4. Prepare the Dataset

Download the MVTec AD dataset and arrange the bottle category according to the directory structure expected by the dataset preparation scripts.

The processed dataset should be available under:

```text
data/processed/bottle/
```

The expected processed structure is:

```text
data/processed/bottle/
|
|-- train/
|   |-- good/
|
|-- test/
|   |-- good/
|   |
|   |-- defective/
|       |-- images/
|       |-- masks/
```

The dataset preparation script is:

```powershell
python src/data/prepare_dataset.py
```

Check the script configuration and dataset source requirements before running it on a fresh environment.

### 5. Create the Training and Calibration Split

```powershell
python src/evaluation/create_calibration_split.py
```

This creates a split containing 167 normal training images and 42 normal calibration images.

### 6. Train the Autoencoder

```powershell
python src/training/train_autoencoder.py
```

### 7. Train PaDiM

```powershell
python src/training/train_padim.py
```

### 8. Train PatchCore

```powershell
python src/training/train_patchcore.py
```

### 9. Calibrate Thresholds

```powershell
python src/evaluation/calibrate_thresholds.py
```

This generates model-specific decision thresholds using normal calibration images.

### 10. Evaluate the Models

Autoencoder:

```powershell
python src/evaluation/evaluate_autoencoder.py
```

PaDiM:

```powershell
python src/evaluation/evaluate_padim.py
```

PatchCore:

```powershell
python src/evaluation/evaluate_patchcore.py
```

### 11. Run the Inference Test

```powershell
python src/inference/test_inference.py
```

### 12. Launch the Application

```powershell
streamlit run app.py
```

Note: A fresh reproduction requires the dataset and trained checkpoints. They are intentionally not included in this GitHub repository.

## Future Improvements

Potential improvements for future development include:

- Evaluate additional MVTec AD categories.
- Experiment with alternative feature extractors.
- Compare different image resolutions.
- Investigate alternative image-level scoring strategies, such as top-k anomaly scores.
- Improve memory-bank selection strategies.
- Evaluate performance under different lighting conditions.
- Test on independently collected industrial images.
- Improve inference efficiency.
- Add batch image inspection.
- Explore deployment in a production-style inspection pipeline.

## Limitations

- The current experiment focuses on the bottle category of MVTec AD.
- The calibration set contains only 42 normal images.
- The test set contains 83 images, so results should not be treated as proof of production-level generalization.
- Image-level decisions use maximum anomaly-map scores, which can be sensitive to isolated high-scoring pixels.
- Heatmap smoothing and visualization normalization affect how results appear visually, but do not improve the underlying model's detection capability.
- Real industrial deployment would require validation under the actual camera, lighting, product, and manufacturing conditions.

## Author

**Mehedi Hasan**

- Electrical and Electronic Engineering
- Bangladesh University of Engineering and Technology
- GitHub: https://github.com/uthaaaan
- Email: mehedihasanrtl23@gmail.com

## Acknowledgements

- MVTec AD dataset for providing an industrial anomaly detection benchmark.
- PyTorch and Torchvision for deep learning tools and pretrained models.
- The open-source computer vision and machine learning community.

---

**ForgeVision - Learning Normality to Detect Industrial Defects.**
