PPG Blood Pressure Estimation Project
This is a PPG signal-based blood pressure estimation project using deep learning models (PaPaGei-S, PaPaGei-S sVRI, PaPaGei-P) for blood pressure prediction.
Project Structure
textproject_root/
├── main.py                    # Main entry file
├── install_dependencies.py    # Dependency installation module
├── preprocess_data.py         # Data preprocessing module
├── train_papagei_s.py         # PaPaGei-S training module (diastolic blood pressure)
├── train_papagei_s_svri.py    # PaPaGei-S sVRI training module (heart rate)
├── train_papagei_p.py         # PaPaGei-P training module (systolic blood pressure)
├── linear_evaluation.py       # Linear evaluation module
├── visualization.py           # Visualization module for training curves, predictions, and reports
├── utils.py                   # Utility functions module
├── models/
│   └── resnet.py              # ResNet model definition
└── README.md                  # Project documentation
Data Requirements
Please ensure your data directory structure is as follows:
textE:\thsiu-ppg\5459299\PPG-BP Database/
├── Data File/
│   ├── PPG-BP dataset.xlsx    # Dataset Excel file
│   ├── 0_subject/             # Contains PPG signal txt files
│   │   ├── 2_1.txt
│   │   ├── 2_2.txt
│   │   ├── 2_3.txt
│   │   ├── 6_1.txt
│   │   └── ...
│   ├── ppg/                   # Preprocessed data (automatically created)
│   ├── train.csv              # Training set split (automatically created)
│   ├── val.csv                # Validation set split (automatically created)
│   └── test.csv               # Test set split (automatically created)
├── trained_models/            # Trained models (automatically created)
└── features/                  # Extracted features (automatically created)
Installation and Running
1. Create Project Directory and Save Code Files
Save all Python files to your project directory, ensuring the models/ subdirectory is created and resnet.py is placed inside it.
2. Run the Full Workflow
bashpython main.py
This will execute the following steps in sequence:

Install dependencies
Preprocess data
Train PaPaGei-S model (diastolic blood pressure prediction)
Train PaPaGei-S sVRI model (heart rate prediction)
Train PaPaGei-P model (systolic blood pressure prediction)
Perform linear evaluation

3. Run Individual Modules
You can also run individual modules:
bash# Install dependencies
python install_dependencies.py

# Data preprocessing
python preprocess_data.py

# Train models
python train_papagei_s.py
python train_papagei_s_svri.py
python train_papagei_p.py

# Evaluation
python linear_evaluation.py
Model Description

PaPaGei-S: Uses ResNet1DMoE architecture for diastolic blood pressure prediction
PaPaGei-S sVRI: Uses ResNet1D architecture for heart rate prediction
PaPaGei-P: Uses ResNet1D architecture for systolic blood pressure prediction

Main Features

Data Preprocessing:

Read PPG signal data
Signal normalization and resampling
Data segmentation and saving


Model Training:

Deep learning model training
Feature extraction and saving
Model saving and loading


Evaluation:

Use Ridge regression for linear evaluation
Calculate MAE, RMSE, and other metrics
Save and display results



Visualization Module
The visualization.py module provides functions to visualize training processes, prediction results, PPG signals, model comparisons, and generate evaluation reports. It uses libraries like Matplotlib, Seaborn, and SciPy for plotting and statistics.
Key functions include:

plot_training_curves(train_losses, val_losses, model_name, save_dir): Plots training and validation loss curves, including standard loss, log-scale loss, loss difference (for overfitting monitoring), and smoothed loss curves. Saves the figure as PNG and PDF.
plot_prediction_results(y_true, y_pred, model_name, target_name, save_dir): Plots prediction vs. truth scatter plot (with regression line), residual plot, residual distribution histogram, and a time-series comparison of the first 100 samples. Includes metrics like MAE, RMSE, and R². Saves the figure as PNG and PDF.
plot_ppg_signal_examples(ppg_signals, predictions, targets, model_name, target_name, save_dir, n_examples=6): Plots waveform examples of PPG signals with corresponding true and predicted values. Color-codes the background based on prediction error (green for low error, orange for medium, red for high). Saves the figure as PNG and PDF.
plot_model_comparison(results_dict, save_dir): Compares performance across models using bar charts for MAE and RMSE. Saves the figure as PNG and PDF.
create_evaluation_report(results_dict, save_dir): Generates an HTML report with a table of model results (MAE, RMSE, R², performance rating), notes on metrics, and performance criteria. Saves the report as evaluation_report.html.

To use this module, import it in your scripts and call the functions with appropriate data (e.g., losses from training, predictions from evaluation). Example usage:
pythonfrom visualization import plot_training_curves, plot_prediction_results  # etc.

# After training
plot_training_curves(train_losses, val_losses, 'PaPaGei-S', 'visualizations/')
Output Results
After running the program, you will see:

Training processes and losses for each model
Final evaluation metrics (MAE, RMSE)
Saved model files and feature files
Visualization outputs (if using visualization.py): PNG/PDF plots of curves, predictions, signals, comparisons, and an HTML evaluation report

Notes

Ensure the data path is correct; update the download_dir variable in files if needed
If GPU is available, the program will automatically use CUDA for accelerated training
Training may take a long time; please be patient
If encountering memory issues, reduce the batch_size parameter

Dependencies
Main dependencies include:

torch, torchvision (PyTorch)
pandas, numpy (data processing)
scikit-learn (machine learning)
tqdm (progress bar)
joblib (data saving)
torch_ecg, pyPPG (signal processing)

Troubleshooting

Import Errors: Ensure all files are in the correct locations and the models directory exists
Data Files Not Found: Check if the data path is correct
CUDA Errors: If no GPU is available, the program will automatically use CPU
Memory Shortage: Reduce batch_size or num_workers param
