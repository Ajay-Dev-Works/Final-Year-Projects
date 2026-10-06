# Financial Fraud Detection Demo

A Streamlit demonstration of transaction preprocessing, an XGBoost fraud classifier, held-out evaluation, local and global SHAP explanations, and a simple rule-based baseline.

## Project structure

```text
final-year-project/
├── app.py
├── requirements.txt
└── README.md
```

## Data

No dataset or trained model was present when this demo was created. With no upload, the app generates a repeatable **synthetic** transaction dataset locally and trains a fresh model. It is for demonstrating the flow, not a claim about real-world performance.

To use your project dataset, upload a labeled CSV from the sidebar. Choose its binary fraud label column when prompted. Rows are split into stratified 75% training and 25% test sets; preprocessing medians and the XGBoost model are fit only on the training set. Numeric columns are used as features. For a Credit Card Fraud Detection-style CSV, `Class` is detected automatically.

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
cd "C:\Users\zenteiq\Desktop\Zenteiq\final-year-project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run the app using the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local address, normally `http://localhost:8501`.

## Exact live demo steps

1. Start the app using the setup commands. Leave the upload empty to use the built-in synthetic data, or upload your labeled CSV.
2. In **Overview**, describe the stratified split and training-only median imputation. Point out the synthetic-data notice if using the fallback.
3. Open **Transaction walkthrough** and select **Highest-scoring test example**. Click **Run detection** to show the model decision, probability, and transaction-level SHAP waterfall.
4. Point out the ranked feature contributions: positive values move the model output toward fraud; negative values move it toward legitimate. SHAP's waterfall is on the raw log-odds scale.
5. Change the starting transaction to **Lowest-scoring test example** and run it again. Optionally select **Custom values** and edit the row in the table before running.
6. Open **Evaluation & comparison**. Explain that Accuracy, Precision, Recall, F1-score, confusion matrix, and global mean-absolute SHAP importance are calculated from held-out rows, not typed-in demo values.
7. Compare the XGBoost and rule-based rows. Adjust the two sidebar thresholds and note how the operating point changes.

The rule baseline flags a transaction when its amount exceeds the selected training-set percentile, or when another feature is an absolute-value outlier at that percentile. Its thresholds are derived from training features without using labels.