# <div align="center">🩺 SVM Diabetes Prediction</div>

<div align="center">
  <h3>🤖 Machine Learning Model for Diabetes Prediction using Support Vector Machine</h3>
  <p><em>Predicting diabetes risk using medical indicators with high accuracy SVM classification</em></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/) [![Machine Learning](https://img.shields.io/badge/ML-Support%20Vector%20Machine-orange.svg)](https://scikit-learn.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-blue.svg)](https://scikit-learn.org/) [![pandas](https://img.shields.io/badge/pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/) [![numpy](https://img.shields.io/badge/numpy-1.21%2B-blue.svg)](https://numpy.org/)

</div>

---

<div align="justify">

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🔬 Dataset Information](#-dataset-information)
- [🚀 Features](#-features)
- [⚙️ Installation](#️-installation)
- [💻 Usage](#-usage)
- [🧠 Machine Learning Approach](#-machine-learning-approach)
- [📊 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [🔍 Technical Implementation](#-technical-implementation)
- [📈 Results & Analysis](#-results--analysis)
- [🛠️ Technologies Used](#️-technologies-used)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👨‍💻 Author](#-author)

## 🎯 Project Overview

This project implements a **Support Vector Machine (SVM)** classifier to predict diabetes in patients based on medical diagnostic measurements. The model analyzes various health indicators to determine the likelihood of diabetes, providing a valuable tool for early diagnosis and preventive healthcare.

### 🎯 Objectives

- Build an accurate diabetes prediction model using SVM
- Analyze medical features that contribute to diabetes risk
- Provide a reliable tool for healthcare screening
- Demonstrate machine learning applications in medical diagnosis

### 🏥 Medical Significance

- **Early Detection**: Helps identify diabetes risk before symptoms appear
- **Preventive Care**: Enables timely intervention and lifestyle modifications
- **Healthcare Efficiency**: Assists medical professionals in screening processes
- **Data-Driven Decisions**: Provides objective risk assessment based on measurable factors

## 🔬 Dataset Information

The project uses the **Pima Indian Diabetes Database**, a well-known medical dataset for diabetes prediction research.

### 📊 Dataset Statistics

- **Total Samples**: 768 patients
- **Features**: 8 medical predictor variables
- **Target Classes**: Binary (Diabetic/Non-Diabetic)
- **Class Distribution**:
  - Non-Diabetic: 500 instances (65.1%)
  - Diabetic: 268 instances (34.9%)

### 🩺 Medical Features

| Feature                      | Description                  | Unit    | Range      |
| ---------------------------- | ---------------------------- | ------- | ---------- |
| **Pregnancies**              | Number of times pregnant     | Count   | 0-17       |
| **Glucose**                  | Plasma glucose concentration | mg/dL   | 0-199      |
| **BloodPressure**            | Diastolic blood pressure     | mmHg    | 0-122      |
| **SkinThickness**            | Triceps skin fold thickness  | mm      | 0-99       |
| **Insulin**                  | 2-Hour serum insulin         | mu U/ml | 0-846      |
| **BMI**                      | Body mass index              | kg/m²   | 0-67.1     |
| **DiabetesPedigreeFunction** | Diabetes pedigree function   | Score   | 0.078-2.42 |
| **Age**                      | Age of patient               | Years   | 21-81      |

### 🎯 Target Variable

- **Outcome**:
  - `0` = Non-Diabetic
  - `1` = Diabetic

## 🚀 Features

### ✨ Core Functionality

- **🔬 Data Preprocessing**: Comprehensive data cleaning and standardization
- **🤖 SVM Classification**: Linear kernel SVM implementation
- **📊 Model Evaluation**: Accuracy assessment on training and testing sets
- **🔮 Prediction System**: Real-time diabetes risk prediction
- **📈 Performance Metrics**: Detailed model performance analysis

### 🛠️ Technical Features

- **Data Standardization**: StandardScaler for feature normalization
- **Train-Test Split**: Stratified sampling for balanced evaluation
- **Cross-Validation Ready**: Extensible for k-fold validation
- **Modular Design**: Clean, reusable code structure

## ⚙️ Installation

### 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Jupyter Notebook (recommended)

### 🔧 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/NhanPhamThanh-IT/SVM-Diabetes-Prediction.git
   cd SVM-Diabetes-Prediction
   ```

2. **Create Virtual Environment** (Recommended)

   ```bash
   python -m venv diabetes_env

   # Windows
   diabetes_env\Scripts\activate

   # macOS/Linux
   source diabetes_env/bin/activate
   ```

3. **Install Required Packages**

   ```bash
   pip install numpy pandas scikit-learn jupyter matplotlib seaborn
   ```

   Or using requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook SVM-Diabetes-Prediction.ipynb
   ```

### 📦 Required Dependencies

```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
jupyter>=1.0.0
matplotlib>=3.4.0  # For visualization (optional)
seaborn>=0.11.0    # For advanced plots (optional)
```

## 💻 Usage

### 🚀 Quick Start

1. **Open the Jupyter Notebook**

   ```bash
   jupyter notebook SVM-Diabetes-Prediction.ipynb
   ```

2. **Run All Cells**
   - Execute cells sequentially to train the model
   - Observe data analysis and preprocessing steps
   - View model training and evaluation results

### 🔮 Making Predictions

The model accepts 8 medical features to predict diabetes risk:

```python
# Example prediction
import numpy as np

# Input format: [Pregnancies, Glucose, BloodPressure, SkinThickness,
#                Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])

# Standardize and predict
std_data = scaler.transform(sample_data)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("🟢 The person is not diabetic")
else:
    print("🔴 The person is diabetic")
```

### 📊 Model Training Process

1. **Data Loading**: Import diabetes dataset
2. **Exploratory Analysis**: Statistical summary and data distribution
3. **Data Preprocessing**: Feature standardization using StandardScaler
4. **Data Splitting**: 80-20 train-test split with stratification
5. **Model Training**: Linear SVM classifier training
6. **Evaluation**: Accuracy calculation on both training and testing sets
7. **Prediction**: Individual risk assessment system

## 🧠 Machine Learning Approach

### 🔬 Support Vector Machine (SVM)

**Why SVM for Diabetes Prediction?**

- **Linear Separability**: Finds optimal decision boundary between classes
- **High Dimensional Data**: Effective with multiple medical features
- **Robust to Overfitting**: Generalizes well on medical datasets
- **Margin Maximization**: Creates reliable classification boundaries

### ⚙️ Model Configuration

```python
# SVM Classifier Setup
classifier = svm.SVC(kernel='linear')

# Key Parameters:
# - kernel='linear': Linear decision boundary
# - C=1.0 (default): Regularization parameter
# - gamma='scale': Kernel coefficient
```

### 🔄 Data Preprocessing Pipeline

1. **Feature Extraction**: Separate features (X) and target (Y)
2. **Standardization**: StandardScaler normalization
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
3. **Train-Test Split**: Stratified 80-20 split
   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(
       X, Y, test_size=0.2, stratify=Y, random_state=2
   )
   ```

## 📊 Model Performance

### 🎯 Accuracy Metrics

The model achieves high accuracy on both training and testing datasets:

- **Training Accuracy**: ~78-80%
- **Testing Accuracy**: ~75-77%
- **Balanced Performance**: Good generalization without overfitting

### 📈 Performance Analysis

**Strengths:**

- ✅ High accuracy on medical diagnostic task
- ✅ Balanced performance across classes
- ✅ Fast prediction capability
- ✅ Interpretable linear decision boundary

**Considerations:**

- 📊 Dataset size limitations (768 samples)
- 🔄 Potential for ensemble improvements
- 📈 Room for hyperparameter optimization

## 📁 Project Structure

```
SVM-Diabetes-Prediction/
│
├── 📓 SVM-Diabetes-Prediction.ipynb   # Main Jupyter notebook
├── 📊 diabetes_data.csv               # Dataset file
├── 📝 README.md                       # Project documentation
├── 📜 LICENSE                         # MIT License
│
├── 📁 docs/                           # Documentation
│   ├── 📋 dataset.md                  # Dataset documentation
│   └── 🧠 svm-algorithm.md            # SVM algorithm details
│
└── 📁 models/                         # Saved models (optional)
    └── 🤖 svm_diabetes_model.pkl      # Trained model
```

## 🔍 Technical Implementation

### 🧩 Code Architecture

**1. Data Import & Analysis**

```python
# Load and explore dataset
diabetes_dataset = pd.read_csv('diabetes_data.csv')
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
```

**2. Feature Engineering**

```python
# Separate features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

**3. Data Standardization**

```python
# Normalize features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
```

**4. Model Training**

```python
# Train SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

**5. Model Evaluation**

```python
# Calculate accuracy
train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))
```

### 🔧 Extensibility

The codebase is designed for easy extension:

- **Additional Kernels**: RBF, polynomial, sigmoid
- **Hyperparameter Tuning**: Grid search implementation
- **Cross-Validation**: K-fold validation
- **Feature Selection**: Recursive feature elimination
- **Ensemble Methods**: Voting classifiers

## 📈 Results & Analysis

### 📊 Key Findings

1. **Model Effectiveness**: SVM demonstrates strong performance for diabetes prediction
2. **Feature Importance**: Glucose level and BMI are strong predictors
3. **Generalization**: Model maintains consistent accuracy across train/test sets
4. **Clinical Relevance**: Results align with medical knowledge of diabetes risk factors

### 🎯 Medical Insights

- **Glucose Levels**: Primary indicator of diabetes risk
- **BMI Correlation**: Strong relationship with diabetes occurrence
- **Age Factor**: Increasing risk with age
- **Family History**: Diabetes pedigree function significance

## 🛠️ Technologies Used

### 🐍 Core Technologies

- **Python**: Primary programming language
- **Jupyter Notebook**: Interactive development environment
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools

### 📚 Libraries & Frameworks

```python
import numpy as np              # Numerical operations
import pandas as pd             # Data manipulation
from sklearn.preprocessing import StandardScaler    # Feature scaling
from sklearn.model_selection import train_test_split # Data splitting
from sklearn import svm         # Support Vector Machine
from sklearn.metrics import accuracy_score          # Model evaluation
```

### 🔧 Development Tools

- **Git**: Version control
- **Markdown**: Documentation
- **CSV**: Data storage format

## 🤝 Contributing

We welcome contributions to improve the diabetes prediction model! Here's how you can contribute:

### 🌟 Ways to Contribute

1. **🐛 Bug Reports**: Report issues or bugs
2. **💡 Feature Requests**: Suggest new features or improvements
3. **📝 Documentation**: Improve project documentation
4. **🔬 Model Enhancement**: Optimize algorithms or add new models
5. **📊 Data Analysis**: Enhance data preprocessing or visualization

### 📋 Contribution Guidelines

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make Changes**
4. **Add Tests** (if applicable)
5. **Commit Changes**
   ```bash
   git commit -m "Add: Your descriptive commit message"
   ```
6. **Push to Branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create Pull Request**

### 🎯 Areas for Improvement

- **Model Optimization**: Hyperparameter tuning, cross-validation
- **Additional Algorithms**: Random Forest, Gradient Boosting, Neural Networks
- **Data Visualization**: Enhanced plots and statistical analysis
- **Web Interface**: Flask/Django web application
- **Mobile App**: Mobile diabetes risk calculator
- **API Development**: REST API for model predictions

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 NhanPhamThanh-IT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👨‍💻 Author

**NhanPhamThanh-IT**

- 🌐 **GitHub**: [@NhanPhamThanh-IT](https://github.com/NhanPhamThanh-IT)
- 📧 **Email**: [ptnhanit230104@gmail.com](ptnhanit230104@gmail.com)

### 🙏 Acknowledgments

- **Dataset Source**: Pima Indian Diabetes Database
- **Inspiration**: Medical machine learning applications
- **Community**: Open source contributors and researchers
- **Libraries**: Scikit-learn, Pandas, NumPy development teams

---

<div align="center">
  <h3>⭐ If you found this project helpful, please give it a star! ⭐</h3>
  <p><em>🩺 Advancing healthcare through machine learning 🤖</em></p>
  
  ![GitHub stars](https://img.shields.io/github/stars/NhanPhamThanh-IT/SVM-Diabetes-Prediction?style=social) ![GitHub forks](https://img.shields.io/github/forks/NhanPhamThanh-IT/SVM-Diabetes-Prediction?style=social) ![GitHub watchers](https://img.shields.io/github/watchers/NhanPhamThanh-IT/SVM-Diabetes-Prediction?style=social)
</div>

</div>

---

<div align="center">
  <sub>Built with ❤️ for healthcare innovation | Last updated: July 2025</sub>
</div>
