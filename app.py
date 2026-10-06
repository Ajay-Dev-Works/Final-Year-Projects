from __future__ import annotations

from io import BytesIO
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Fraud Detection Lab",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #142b2a; --muted: #607270; --green: #087e68; --line: #dce7e3; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0 !important; }
    .stApp { background: linear-gradient(145deg, #f5faf7 0%, #edf5f1 54%, #f8faf7 100%); }
    [data-testid="stSidebar"] { background: #e7f1ec; border-right: 1px solid var(--line); }
    .hero { padding: 1.1rem 0 1.4rem; border-bottom: 1px solid var(--line); margin-bottom: 1.4rem; }
    .eyebrow { color: var(--green); text-transform: uppercase; font-weight: 700; font-size: .76rem; letter-spacing: .08em; }
    .subtle { color: var(--muted); font-size: 1rem; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); padding: .85rem 1rem; border-radius: 6px; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .stButton button[kind="primary"] { background: var(--green); border-color: var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_demo_data(rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate a repeatable, clearly synthetic dataset for an offline demo."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(rows) < 0.035
    values = {f"V{i}": rng.normal(0, 1, rows) for i in range(1, 9)}
    shifts = {"V1": -2.5, "V2": 1.8, "V3": -2.2, "V4": 2.0, "V5": -1.5}
    for feature, shift in shifts.items():
        values[feature] += fraud * shift
    values["Time"] = rng.uniform(0, 172800, rows)
    values["Amount"] = np.where(
        fraud,
        rng.lognormal(mean=4.0, sigma=1.0, size=rows),
        rng.lognormal(mean=3.0, sigma=1.0, size=rows),
    )
    return pd.DataFrame(values).assign(Class=fraud.astype(int))


def positive_labels(target: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(target, errors="coerce")
    classes = pd.unique(target.dropna())
    if len(classes) != 2:
        raise ValueError("The selected target column must contain exactly two classes.")
    if set(pd.unique(numeric.dropna())) == {0, 1}:
        return numeric.eq(1).where(numeric.notna())

    normalized = pd.Series(target.astype(str).str.strip().str.lower(), index=target.index)
    positive_terms = {"fraud", "fraudulent", "1", "yes", "true", "positive"}
    positive_classes = [value for value in classes if str(value).strip().lower() in positive_terms]
    if positive_classes:
        return target.eq(positive_classes[0]).where(target.notna())
    counts = target.value_counts(dropna=True)
    minority_class = str(counts.idxmin())
    return target.astype(str).eq(minority_class).where(target.notna())


@st.cache_resource(show_spinner="Training and evaluating the XGBoost model...")
def train_model(dataframe: pd.DataFrame, target_column: str) -> dict:
    labels = positive_labels(dataframe[target_column])
    features = dataframe.drop(columns=[target_column]).copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")

    valid_rows = labels.notna()
    features = features.loc[valid_rows]
    labels = labels.loc[valid_rows].astype(int)
    if features.empty or features.shape[1] == 0:
        raise ValueError("No numeric feature columns were found after preprocessing.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    medians = X_train.median().fillna(0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=negatives / max(positives, 1),
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "explainer": explainer,
        "features": list(features.columns),
        "medians": medians,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_probabilities": pd.Series(test_probabilities, index=X_test.index),
        "fraud_rate": float(labels.mean()),
        "class_counts": labels.value_counts().sort_index(),
    }


def get_shap_values(explainer: shap.TreeExplainer, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(frame)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_value = np.asarray(explainer.expected_value).reshape(-1)
    return values, float(base_value[-1])


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": cast(float, accuracy_score(actual, predicted)),
        "Precision": cast(float, precision_score(actual, predicted, zero_division=0)),
        "Recall": cast(float, recall_score(actual, predicted, zero_division=0)),
        "F1-score": cast(float, f1_score(actual, predicted, zero_division=0)),
    }


def rule_predictions(
    frame: pd.DataFrame, reference: pd.DataFrame, percentile: float
) -> tuple[np.ndarray, list[str], list[float]]:
    amount_names = {"amount", "transactionamount", "transaction_amount", "amt"}
    amount_column = next(
        (column for column in frame.columns if column.lower() in amount_names), None
    )
    flags = np.zeros(len(frame), dtype=bool)
    descriptions: list[str] = []
    thresholds: list[float] = []

    if amount_column is not None:
        amount_limit = float(reference[amount_column].quantile(percentile))
        amount_flags = frame[amount_column].gt(amount_limit).fillna(False).to_numpy()
        flags |= amount_flags
        descriptions.append(f"Amount exceeds the training-set {percentile:.1f}th percentile")
        thresholds.append(amount_limit)

    extreme_columns = [column for column in frame.columns if column != amount_column]
    if extreme_columns:
        limits = reference[extreme_columns].abs().quantile(percentile)
        extreme_flags = frame[extreme_columns].abs().gt(limits).fillna(False).any(axis=1).to_numpy()
        flags |= extreme_flags
        descriptions.append(
            f"Any other feature exceeds its training-set {percentile:.1f}th absolute percentile"
        )
        thresholds.extend(float(value) for value in limits.dropna().values)
    return flags.astype(int), descriptions, thresholds


def show_metric_row(metrics: dict[str, float]) -> None:
    columns = st.columns(4)
    for column, name in zip(columns, metrics):
        column.metric(name, f"{metrics[name]:.3f}")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Final-year project · Model demonstration</div>
      <h1 style="margin:.35rem 0">Financial Fraud Detection</h1>
      <div class="subtle">XGBoost prediction, SHAP explanation, and a transparent rules baseline.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data & settings")
    uploaded_file = st.file_uploader("Upload a labeled CSV", type=["csv"])
    if uploaded_file is None:
        dataframe = make_demo_data()
        st.caption("Using generated demo data. It is synthetic, not real bank records.")
        target_column = "Class"
    else:
        try:
            dataframe = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")
            st.stop()
        if dataframe.empty:
            st.error("The uploaded CSV has no rows.")
            st.stop()
        likely_targets = [
            column for column in dataframe.columns
            if column.strip().lower() in {"class", "fraud", "is_fraud", "label", "target"}
        ]
        default_target = likely_targets[0] if likely_targets else dataframe.columns[-1]
        default_index = list(dataframe.columns).index(default_target)
        target_column = st.selectbox(
            "Which column is the fraud label?",
            options=list(dataframe.columns),
            index=default_index,
        )
        st.caption("Only numeric input columns are used; missing values are filled from training medians.")

    threshold = st.slider("XGBoost fraud decision threshold", 0.05, 0.95, 0.50, 0.01)
    rule_percentile = st.slider("Rule-based outlier percentile", 95.0, 99.9, 99.5, 0.1)

try:
    artifacts = train_model(dataframe, target_column)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.info("Use a CSV with a binary fraud label and numeric transaction columns.")
    st.stop()

model = artifacts["model"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]
test_probabilities = artifacts["test_probabilities"]
model_test_predictions = (test_probabilities >= threshold).astype(int)
model_metrics = calculate_metrics(y_test, model_test_predictions)
rules_test_predictions, rule_descriptions, rule_thresholds = rule_predictions(
    X_test, artifacts["X_train"], rule_percentile / 100
)
rule_metrics = calculate_metrics(y_test, rules_test_predictions)

overview, input_tab, evaluation_tab = st.tabs(
    ["Overview", "Transaction walkthrough", "Evaluation & comparison"]
)

with overview:
    st.subheader("The detection flow")
    st.markdown(
        "**Transaction input** → **numeric preprocessing** → **XGBoost** → "
        "**fraud probability** → **SHAP explanation** → **rule-based comparison**"
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Dataset and model")
        st.write(
            f"**Rows:** {len(dataframe):,}  ·  **Features used:** {len(artifacts['features'])}  ·  "
            f"**Fraud rate:** {artifacts['fraud_rate']:.2%}"
        )
        st.write(
            f"**Evaluation:** stratified 75/25 train/test split  ·  "
            f"**Test transactions:** {len(X_test):,}"
        )
        st.write("**Preprocessing:** numeric conversion, infinite-value cleanup, then training-median imputation.")
    with right:
        st.markdown("#### What the baseline checks")
        if rule_descriptions:
            for description in rule_descriptions:
                st.write(f"- {description}")
            st.caption("Thresholds are learned from the training features only; no labels are used by the rules.")
        else:
            st.write("No eligible input features were available for the rule baseline.")

with input_tab:
    st.subheader("Score one transaction")
    st.caption("Choose a held-out example or edit its numeric values. The selected label is not shown to the model.")
    choices = ["Highest-scoring test example", "Lowest-scoring test example", "Custom values"]
    choice = st.selectbox("Starting transaction", choices)
    if choice == choices[0]:
        starting_row = X_test.loc[[test_probabilities.idxmax()]].copy()
    elif choice == choices[1]:
        starting_row = X_test.loc[[test_probabilities.idxmin()]].copy()
    else:
        starting_row = pd.DataFrame([artifacts["medians"]], columns=artifacts["features"])

    edited_row = st.data_editor(
        starting_row,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        key=f"transaction-input-{choice}",
        column_config={
            column: st.column_config.NumberColumn(column, format="%.5f")
            for column in artifacts["features"]
        },
    )
    if st.button("Run detection", type="primary", icon="🔍"):
        input_values = edited_row.reindex(columns=artifacts["features"])
        input_values = input_values.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        input_values = input_values.fillna(artifacts["medians"])
        fraud_probability = float(model.predict_proba(input_values)[0, 1])
        prediction = int(fraud_probability >= threshold)
        label = "FRAUD" if prediction else "LEGITIMATE"

        st.divider()
        result_col, probability_col, threshold_col = st.columns(3)
        result_col.metric("XGBoost decision", label)
        probability_col.metric("Fraud probability", f"{fraud_probability:.2%}")
        threshold_col.metric("Decision threshold", f"{threshold:.0%}")

        local_values, base_value = get_shap_values(artifacts["explainer"], input_values)
        shap_explanation = shap.Explanation(
            values=local_values[0],
            base_values=base_value,
            data=input_values.iloc[0].to_numpy(),
            feature_names=artifacts["features"],
        )
        shap_col, reason_col = st.columns([1.5, 1])
        with shap_col:
            st.markdown("#### Why did the model decide this?")
            st.caption("SHAP contributions are on the model's raw log-odds scale, not probability points.")
            shap.plots.waterfall(shap_explanation, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True, width="stretch")
            plt.close("all")
        with reason_col:
            st.markdown("#### Largest transaction contributions")
            contribution_table = pd.DataFrame(
                {"Feature": artifacts["features"], "SHAP contribution": local_values[0]}
            )
            contribution_table["Direction"] = np.where(
                contribution_table["SHAP contribution"] >= 0,
                "Pushes toward fraud",
                "Pushes toward legitimate",
            )
            contribution_table["Magnitude"] = contribution_table["SHAP contribution"].abs()
            st.dataframe(
                contribution_table.sort_values("Magnitude", ascending=False)
                .drop(columns="Magnitude")
                .head(8),
                hide_index=True,
                width="stretch",
            )

        input_rules, _, _ = rule_predictions(
            input_values, artifacts["X_train"], rule_percentile / 100
        )
        rule_label = "FRAUD ALERT" if input_rules[0] else "NO ALERT"
        st.info(f"Rule-based result: **{rule_label}**. Rules flag outlier amounts or extreme feature values.")

with evaluation_tab:
    st.subheader("Held-out test results")
    st.caption(
        f"All metrics below are calculated from {len(y_test):,} unseen test rows. "
        "Adjust the threshold in the sidebar to see its effect on XGBoost metrics."
    )
    st.markdown("#### XGBoost")
    show_metric_row(model_metrics)
    matrix_col, comparison_col = st.columns([1, 1.2])
    with matrix_col:
        st.markdown("#### Confusion matrix")
        model_matrix = confusion_matrix(y_test, model_test_predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4.4, 3.4))
        ConfusionMatrixDisplay(
            confusion_matrix=model_matrix,
            display_labels=["Legitimate", "Fraud"],
        ).plot(ax=axis, cmap="Greens", colorbar=False, values_format="d")
        axis.set_title("XGBoost · test set")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with comparison_col:
        st.markdown("#### Rules vs. XGBoost")
        comparison = pd.DataFrame(
            [model_metrics, rule_metrics],
            index=["XGBoost", "Traditional rules"],
        )
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        rule_matrix = confusion_matrix(y_test, rules_test_predictions, labels=[0, 1])
        tn, fp, fn, tp = rule_matrix.ravel()
        st.caption(f"Rule confusion matrix: TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}")

    st.markdown("#### Global SHAP feature importance")
    shap_rows = X_test.sample(min(300, len(X_test)), random_state=42)
    global_values, _ = get_shap_values(artifacts["explainer"], shap_rows)
    importance = pd.DataFrame(
        {"Feature": artifacts["features"], "Mean |SHAP|": np.abs(global_values).mean(axis=0)}
    ).sort_values("Mean |SHAP|", ascending=True).tail(12)
    st.bar_chart(importance.set_index("Feature"), horizontal=True, x_label="Mean absolute SHAP value")
    st.caption("Global importance is mean absolute SHAP across a sample of held-out transactions.")
