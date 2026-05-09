# 🏡 Lab 01: House Price Prediction via Linear Regression

## 📝 Overview
This project is the first assignment for the **Mathematical Methods for Artificial Intelligence** course. The objective is to build a Linear Regression model to predict house prices (`SalePrice`) based on various house features. The highlight of this project is **implementing mathematical optimization algorithms entirely from scratch**, without relying on high-level Machine Learning libraries, to gain a deep understanding of the model's core mechanics.

## ✨ Technical Highlights
- **From Scratch Implementation:** Deployed the **Gradient Descent** algorithm entirely using `NumPy` for matrix calculations and weight updates.
- **Exploratory Data Analysis (EDA):** Utilized `pandas`, `matplotlib`, and `seaborn` to visualize and analyze the distribution of the target variable and the correlation matrix among features.
- **Outlier Handling:** Delved into Error Analysis by visualizing the discrepancy between actual and predicted values (Actual vs Predicted). Conducted detailed comparisons of model variants (Model 2a, 2b, 2c) to evaluate the impact of anomalous data points.
- **Model Evaluation:** Assessed the algorithm's reliability using custom-built metrics and visual analysis of residuals.

## 📂 Directory Structure
- `dataset/`: Contains the training data (`train.csv`), testing data (`test.csv`), and the data dictionary (`data_description.txt`).
- `24122024.ipynb`: The main Jupyter Notebook containing the full source code: from data cleaning and EDA to algorithm implementation and model evaluation.
- `report.pdf`: Comprehensive report detailing the mathematical foundations, loss function derivatives, and experimental results analysis.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TranTrongTu/hcmus-math-for-ai.git](https://github.com/TranTrongTu/hcmus-math-for-ai.git)
   cd hcmus-math-for-ai/lab01
   ```

2. **Environment Setup:**
   Ensure Python is installed. Then, install the required dependencies via the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   Open the notebook to view the training steps and analytical plots:
   ```bash
   jupyter notebook 24122024.ipynb
   ```

## 📈 Results Summary
The model demonstrated stable convergence through the custom-implemented Gradient Descent algorithm. The Actual vs Predicted cross-referencing, combined with isolating outliers, showed that refining the mathematical loss function and optimizing features played a crucial role in improving the model's prediction accuracy.