from pathlib import Path
import pickle
import html

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "linear_svm_model.pkl"

st.set_page_config(
    page_title="NEO Issue Category Predictor",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f5f7fb 0%, #eef2f8 100%); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.hero {
  padding: 1.6rem 1.8rem; border-radius: 18px;
  background: linear-gradient(120deg, #102a43, #243b53, #334e68);
  color: white; margin-bottom: 1.2rem;
  box-shadow: 0 10px 28px rgba(16,42,67,.14);
}
.hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
.hero p { color: #d9e2ec; margin: 0; font-size: 1rem; }
.result-card {
  padding: 1.25rem 1.4rem; border: 1px solid #d9e2ec;
  border-radius: 16px; background: white;
  box-shadow: 0 5px 18px rgba(16,42,67,.06);
}
.result-label { color: #627d98; font-size: .8rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em; }
.result-value { color: #102a43; font-size: 1.65rem; font-weight: 750; }
div.stButton > button { border-radius: 10px; font-weight: 700; min-height: 2.8rem; }
[data-testid="stSidebar"] { background: #ffffff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained ML model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "linear_svm_model.pkl was not found. Place it in the same folder as app.py."
        )
    # Only load pickle files from a trusted source.
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_categories(model, descriptions):
    cleaned = [str(value).strip() for value in descriptions]
    return model.predict(cleaned)


with st.sidebar:
    st.markdown("## 🛠️ NEO Quality Tools")
    st.caption("ML assistant for issue categorization")
    st.divider()
    st.markdown("### How to use")
    st.markdown(
        "1. Enter an issue description or upload a CSV.\n"
        "2. Click **Predict Category**.\n"
        "3. Review the prediction before updating the log book."
    )
    st.info("Predictions are suggestions. An engineer should verify unusual or new issues.")

st.markdown("""
<div class="hero">
  <h1>NEO Issue Category Predictor</h1>
  <p>Predict <b>Issue Category (Component wise)</b> from an issue description
  using your trained TF-IDF + Linear SVM pipeline.</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.markdown(
        "Required files in one folder:\n"
        "```text\n"
        "your_project/\n"
        "├── app.py\n"
        "├── linear_svm_model.pkl\n"
        "└── requirements.txt\n"
        "```"
    )
    st.stop()

single_tab, batch_tab = st.tabs(["✍️ Single Issue", "📄 Batch CSV Prediction"])

with single_tab:
    st.subheader("Describe the issue")
    description = st.text_area(
        "Issue Description",
        placeholder="Example: During testing, the slider belt was found broken near the pulley...",
        height=150,
        help="Enter a clear, specific description of the observed issue.",
    )
    if st.button("🔍 Predict Category", type="primary", use_container_width=True):
        if not description.strip():
            st.warning("Please enter an issue description first.")
        else:
            try:
                prediction = str(predict_categories(model, [description])[0])
                safe_prediction = html.escape(prediction)
                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-label">Predicted issue category</div>'
                    f'<div class="result-value">{safe_prediction}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Verify the suggested label before updating the official production log.")
                result_df = pd.DataFrame({
                    "Issue Description": [description],
                    "Predicted Issue Category (Component wise)": [prediction],
                })
                st.download_button(
                    "⬇️ Download prediction as CSV",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="neo_issue_prediction.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.subheader("Predict categories for multiple issues")
    st.write("Upload a CSV containing a column named **Issue Description**.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "Issue Description" not in df.columns:
                st.error(
                    "Column **Issue Description** was not found. Available columns: "
                    + ", ".join(map(str, df.columns))
                )
            else:
                st.write("Preview of uploaded data")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("⚡ Predict All Categories", type="primary", use_container_width=True):
                    descriptions = df["Issue Description"].fillna("").astype(str)
                    valid_mask = descriptions.str.strip().ne("")
                    df["Predicted Issue Category (Component wise)"] = ""
                    if valid_mask.any():
                        predictions = predict_categories(model, descriptions.loc[valid_mask].tolist())
                        df.loc[valid_mask, "Predicted Issue Category (Component wise)"] = predictions
                    st.success(f"Finished. Predicted {int(valid_mask.sum())} non-empty issue descriptions.")
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download results CSV",
                        data=df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="neo_issue_categories_predicted.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as exc:
            st.error(f"Could not read or process this CSV: {exc}")

st.divider()
st.caption("NEO Issue Category Predictor • Saved TF-IDF + LinearSVC pipeline")
