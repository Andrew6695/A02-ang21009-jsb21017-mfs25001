# Title: A02-ang21009-jsb21017-mfs25001

# California Housing MLP Regression Pipeline

Specifically, the pipeline:
- Loads and inspects the California Housing dataset
- Splits the data into training and testing sets
- Scales features using standardization
- Trains an MLP neural network for regression
- Evaluates performance using R-squared and MAE
- Saves scatter plots comparing actual vs. predicted values

## Files in This Repository
- src/
  - ds_pipeline.py
- figures/
  - train_actual_vs_pred.png
  - test_actual_vs_pred.png
- README.md

## How to Run the Project (Within VS Code)

### 1. Install dependencies
Make sure you have Python 3 installed, then install the required packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Run the script
From the project directory, run:
```bash
python ds_pipeline.py
```

3. Output

- Model performance metrics (R-squared and MAE) will print to the console
- Two scatter plots will be saved in the figures/ folder:
  - train_actual_vs_pred.png
  - test_actual_vs_pred.png
- These plots visualize predicted vs. actual median house values for both training and test data.

## Partners:
- Joseph Berkowitz
- Anh Hao Huynh
- Andrew Ghali
