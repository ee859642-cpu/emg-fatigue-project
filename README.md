# emg-fatigue-project
Integrative project for EMG-based muscle fatigue classification using Machine Learning.
# EMG-Based Detection and Classification of Quadriceps Muscular Fatigue

**Integrative Project - Biomechanics of Gait (Yachay Tech)**
**Group:** [GROUP 3]
**Members:** [CHELSY PRISCILA QUEVEDO DE PRADA], [MAYERLIN ALEJANDRA SOLORZANO ZAMBRANO], [ERICK ALEXANDER YAMBERLA DIAZ]
**Dataset:** 15 subjects, Rectus femoris, dynamic knee extension (70% 1RM).

## Project Overview
This project builds a machine learning pipeline to classify peripheral muscle fatigue from surface EMG signals. We use a temporal-thirds labelling strategy (first third = non-fatigued, last third = fatigued) and extract 13 features (time, frequency, and Hjorth parameters). We validate our models using **Leave-One-Subject-Out (LOSO)** cross-validation to prevent data leakage.

## Repository Structure
- `src/preprocessing.py`: Band-pass (20-450 Hz) and Notch (60 Hz) filters.
- `src/labeling.py`: Temporal-thirds labelling strategy.
- `src/features.py`: Extraction of the 13 features.
- `src/train_model.py`: Training and Nested LOSO validation of Baseline (LDA) and Random Forest models.
- `src/predict.py`: Script to predict fatigue on a new, unseen EMG signal.
- `results.csv`: Final metrics (Mean ± SD across LOSO folds).

## How to Reproduce
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the raw data `.csv` files are in the local directory.
3. Run the training pipeline:
   ```bash
   python src/train_model.py
