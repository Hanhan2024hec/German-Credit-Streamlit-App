import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
from catboost import CatBoostClassifier


st.set_page_config(page_title="German Credit Risk ", page_icon= "🎯", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Explore", "Model", "About"])

train = pd.read_csv("german_credit_train.csv").drop_duplicates()

def proportions_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    counts = df.groupby([by, "Risk"]).size().reset_index(name="count")
    totals = df.groupby(by).size().reset_index(name="total")
    out = counts.merge(totals, on=by)
    out["proportion"] = out["count"] / out["total"]
    return out

def render_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["No risk", "Risk"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    st.pyplot(fig)

train['Risk'] = train['Risk'].map({'No Risk': 0, 'Risk': 1}).astype('Int64')
numeric_cols = ['LoanDuration', 'LoanAmount', 'Age',
                'InstallmentPercent', 'CurrentResidenceDuration']
category_cols = ['CheckingStatus', 'CreditHistory', 'LoanPurpose',
                 'ExistingSavings', 'EmploymentDuration', 'OthersOnLoan',
                 'Sex', 'OwnsProperty', 'InstallmentPlans', 'Housing',
                 'ExistingCreditsCount', 'Job', 'Dependents', 'Telephone']

X = train.drop(['Risk', 'ForeignWorker'], axis=1)
y = train['Risk']
for c in category_cols:
    if c in X.columns:
        X[c] = X[c].astype('string')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols_in_X = [c for c in category_cols if c in X.columns]

# -------------------- PAGES --------------------
if page == "Overview":
    st.title("📊 German Credit Risk")
    st.markdown(
        """
        This app predicts whether a customer is **Risk** or **No Risk** for a lending system,
        based on features like **Gender**, **Age**, **Credit History**, **Employment Duration**, and more.
        
        **Model**: I used **CatBoostClassifier** for robust categorical handling and strong baseline performance.  
        **Evaluation**: Confusion matrix + precision/recall/F1, with an adjustable decision threshold.
        """)
    st.markdown("### Sample of the training data")
    st.dataframe(train.head(20), use_container_width=True)
    st.caption(f"Rows: {len(train):,}  •  Columns: {len(train.columns)}")

elif page == "Explore":
    st.title("🔎 Explore Risk Distributions")
    candidates = ["Sex", "CreditHistory", "EmploymentDuration"]
    by = st.selectbox("Choose a categorical feature", candidates, index=0)
    out = proportions_by(train, by)
    st.subheader(f"Aggregated proportions by {by}")
    st.dataframe(out, use_container_width=True)

    st.subheader(f"Bar chart — Proportion of Risk vs No Risk by {by}")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=out, x=by, y="proportion", hue="Risk", ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(by)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# train_counts = (
#     train.groupby(["Sex", "Risk"])
#       .size()
#       .reset_index(name="count")
#     .merge(train.groupby("Sex").size().reset_index(name="total"), on="Sex")
# )
# train_counts["proportion"] = train_counts["count"] / train_counts["total"]

# # by sex
# st.header("Partition by Sex")
# st.subheader("(1) Aggregated proportions by Sex")
# st.dataframe(train_counts, use_container_width=True)

# st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by Sex")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(data=train_counts, x="Sex", y="proportion", hue="Risk", ax=ax)
# ax.set_title("Proportion of Risk vs No Risk by Sex")
# ax.set_ylabel("Proportion")
# ax.set_ylim(0, 1)
# st.pyplot(fig)

# # by credit history

# train_counts_history = (
#     train.groupby(["CreditHistory", "Risk"])
#       .size()
#       .reset_index(name="count")
#     .merge(train.groupby("CreditHistory").size().reset_index(name="total"), on="CreditHistory")
# )
# train_counts_history["proportion"] = train_counts_history["count"] / train_counts_history["total"]

# st.header("Partition by CreditHistory")
# st.subheader("(1) Aggregated proportions by CreditHistory")
# st.dataframe(train_counts_history, use_container_width=True)

# st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by CreditHistory")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(data=train_counts_history, x="CreditHistory", y="proportion", hue="Risk", ax=ax)
# ax.set_title("Proportion of Risk vs No Risk by CreditHistory")
# ax.set_ylabel("Proportion")
# ax.set_ylim(0, 1)
# st.pyplot(fig)

# # by EmploymentDuration

# train_counts_duration = (
#     train.groupby(["EmploymentDuration", "Risk"])
#       .size()
#       .reset_index(name="count")
#     .merge(train.groupby("EmploymentDuration").size().reset_index(name="total"), on="EmploymentDuration")
# )
# train_counts_duration["proportion"] = train_counts_duration["count"] / train_counts_duration["total"]

# st.header("Partition by EmploymentDuration")
# st.subheader("(1) Aggregated proportions by EmploymentDuration")
# st.dataframe(train_counts_duration, use_container_width=True)

# st.subheader("(2) Bar chart — Proportion of Risk vs No Risk by EmploymentDuration")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(data=train_counts_duration, x="EmploymentDuration", y="proportion", hue="Risk", ax=ax)
# ax.set_title("Proportion of Risk vs No Risk by EmploymentDuration")
# ax.set_ylabel("Proportion")
# ax.set_ylim(0, 1)
# st.pyplot(fig)


elif page == "Model":
    st.title("🤖 Train & Evaluate (CatBoost)")

    with st.sidebar:
        st.subheader("Model Controls")
        iterations = st.slider("Iterations", 100, 1000, 500)
        depth = st.select_slider("Depth", options=[4, 6, 8], value=8)
        lr = st.select_slider("Learning rate", options=[0.005, 0.01, 0.05], value=0.01)
        l2 = st.select_slider("L2 regularization", options=[1, 3, 5], value=3)
        threshold = st.slider("Decision threshold", 0.50, 0.95, 0.75)

    with st.spinner("Training CatBoost..."):
        model_cat = CatBoostClassifier(
        iterations=500,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=l2,
        cat_features=cat_cols_in_X,
        verbose=False,
        random_seed=100
        )
        model_cat.fit(X_train, y_train)


    y_proba = model_cat.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_custom)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_custom, average="binary", zero_division=0)
    # st.subheader("Confusion Matrix")
    # st.metric("Validation Accuracy", f"{acc:.3f}")

    st.subheader("Key metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{precision:.3f}")
    m3.metric("Recall", f"{recall:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")

    st.subheader("Detailed classification report")
    st.code(classification_report(y_test, y_pred_custom, target_names=["No risk", "Risk"]), language="text")

    st.subheader("Confusion matrix")
    render_confusion_matrix(y_test, y_pred_custom, f"Threshold = {threshold:.2f}")

    st.caption("Tip: Adjust the decision threshold to balance precision and recall.")

elif page == "About":
    st.title("ℹ️ About this project")
    st.markdown(
        """
        **Goal**: Predict whether a customer is *Risk* or *No Risk* for a bank lending system  
        using features like **Gender**, **Age**, **Credit History**, **Employment Duration**, etc.

        **Why CatBoost?**  
        - Handles categorical variables natively  
        - Strong performance with minimal preprocessing  
 
        """
    )
    st.markdown("---")
    st.markdown("**Repo**: https://github.com/Hanhan2024hec/Streamlit-Project")
    st.markdown("**Docker Hub**: http://localhost:8600/")

