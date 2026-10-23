from pathlib import Path
import html
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="ResolveAI | Complaint Intelligence",
    page_icon="💠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "models" / "complaint_classifier.joblib"

NAMES = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages and Loans",
    "retail_banking": "Retail Banking",
}

COLORS = ["#087E8B", "#36A3B2", "#7086CA", "#78BFA7", "#A18BC5"]

# Apply all styling before displaying the interface.
st.markdown(
    """
<style>
.stApp {
    background: #EDF6FA;
    color: #173D48;
}

[data-testid="stHeader"] {
    background: #EDF6FA;
}

/* Leave enough room below Streamlit's toolbar. */
.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1320px;
    padding-top: 6rem !important;
    padding-bottom: 2rem !important;
}

h1, h2, h3, h4 {
    color: #124E59 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p {
    color: #294F5C;
}

/* The brand uses regular block layout to avoid clipping. */
.resolve-header {
    display: block;
    padding: 14px 0 22px;
    margin: 0;
    overflow: visible;
}

.resolve-name {
    display: block;
    color: #124E59;
    font-family: Arial, sans-serif;
    font-size: 34px;
    font-weight: 750;
    line-height: 1.6;
    padding: 0;
    margin: 0;
    overflow: visible;
}

.resolve-name span {
    color: #087E8B;
}

.resolve-subtitle {
    color: #60808D;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 0;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D6E8EE;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 16px rgba(18, 78, 89, 0.035);
}

[data-testid="stMetricLabel"] p {
    color: #60808D !important;
    font-size: 13px;
}

[data-testid="stMetricValue"] {
    color: #087E8B;
    font-size: 30px;
}

[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #D6E8EE;
    border-radius: 18px;
    padding: 22px;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #173D48 !important;
    border-color: #C6DFE7 !important;
    border-radius: 10px;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: #173D48 !important;
    -webkit-text-fill-color: #173D48 !important;
}

[data-baseweb="select"] svg {
    fill: #087E8B !important;
}

[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"] {
    background: #FFFFFF !important;
    color: #173D48 !important;
}

[data-testid="stTextArea"] textarea {
    background: #F8FCFE !important;
    color: #173D48 !important;
    -webkit-text-fill-color: #173D48 !important;
    caret-color: #087E8B;
    border-radius: 12px;
    font-size: 16px;
    line-height: 1.7;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #78949F !important;
    -webkit-text-fill-color: #78949F !important;
    opacity: 1;
}

div.stFormSubmitButton > button {
    background: #087E8B !important;
    color: #FFFFFF !important;
    border: 1px solid #087E8B !important;
    border-radius: 10px;
    min-height: 46px;
}

div.stFormSubmitButton > button p {
    color: #FFFFFF !important;
    font-weight: 600;
}

div.stFormSubmitButton > button:hover {
    background: #056571 !important;
}

[data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a {
    background: #FFFFFF;
    color: #087E8B;
    border: 1px solid #BDDCE5;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    margin: 22px 0 16px;
    border-bottom: 1px solid #D6E8EE;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    padding: 12px 18px;
    border-radius: 10px 10px 0 0;
}

.stTabs [data-baseweb="tab"] p {
    color: #60808D !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #DCEFF3 !important;
}

.stTabs [aria-selected="true"] p {
    color: #087E8B !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #087E8B !important;
}

[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #D6E8EE;
    border-radius: 16px;
}

.result-card {
    background: #E0F3F0;
    border: 1px solid #B7DED7;
    border-radius: 15px;
    padding: 20px 24px;
    margin: 10px 0 18px;
}

.result-card small {
    color: #467E78;
    letter-spacing: 1px;
    font-weight: 600;
}

.result-card h3 {
    color: #096C65 !important;
    margin: 8px 0 0;
    line-height: 1.5;
}

.empty-result {
    background: #FFFFFF;
    border: 1px dashed #BBD7E2;
    border-radius: 16px;
    padding: 30px 26px;
    margin-top: 10px;
}

.empty-result h3 {
    font-size: 22px;
    line-height: 1.5;
}

.empty-result p {
    color: #60808D !important;
    line-height: 1.7;
}

.resolve-footer {
    border-top: 1px solid #D6E8EE;
    padding-top: 18px;
    margin-top: 30px;
    color: #60808D;
    font-size: 12px;
}

@media (max-width: 700px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 5rem !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .resolve-name {
        font-size: 29px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports():
    with open(
        REPORT_DIR / "metrics.json",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    report = pd.read_csv(
        REPORT_DIR / "classification_report.csv",
        index_col=0,
    )

    matrix = pd.read_csv(
        REPORT_DIR / "confusion_matrix.csv",
        index_col=0,
    )

    return metrics, report, matrix


def show_chart(fig, left=40, bottom=60, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#294F5C",
        ),
        margin=dict(
            l=left,
            r=45,
            t=60,
            b=bottom,
            autoexpand=True,
        ),
        height=height,
        title=dict(
            x=0.04,
            font=dict(size=17, color="#124E59"),
        ),
        legend=dict(font=dict(color="#294F5C")),
    )

    fig.update_xaxes(
        automargin=True,
        gridcolor="#EAF2F5",
        zerolinecolor="#D6E8EE",
    )

    fig.update_yaxes(
        automargin=True,
        gridcolor="#EAF2F5",
        zerolinecolor="#D6E8EE",
    )

    st.plotly_chart(
        fig,
        theme=None,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


required_files = [
    MODEL_PATH,
    REPORT_DIR / "metrics.json",
    REPORT_DIR / "classification_report.csv",
    REPORT_DIR / "confusion_matrix.csv",
]

if not all(path.exists() for path in required_files):
    st.error("Model files are missing. Run train_model.py first.")
    st.stop()

model = load_model()
metrics, report, matrix = load_reports()
classes = list(model.classes_)

color_map = {
    NAMES.get(label, label): COLORS[index % len(COLORS)]
    for index, label in enumerate(classes)
}

st.markdown(
    """
<div class="resolve-header">
    <div class="resolve-name"><span>✦</span> ResolveAI</div>
    <div class="resolve-subtitle">
        Financial complaint intelligence
    </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
c2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
c3.metric("Training complaints", f"{metrics['training_rows']:,}")
c4.metric("Product categories", len(classes))

studio, analytics, performance, about = st.tabs(
    ["Classify", "Analytics", "Performance", "About"]
)

with studio:
    st.subheader("Understand a complaint in seconds")
    st.caption(
        "Enter a financial complaint to receive a suggested product category."
    )

    input_col, result_col = st.columns(
        [1, 1],
        gap="large",
    )

    examples = {
        "Write my own complaint": "",
        "Credit report problem": (
            "An account that does not belong to me appears on my "
            "credit report and I want it investigated."
        ),
        "Debt collection calls": (
            "A debt collector keeps calling me about a debt "
            "that I already paid in full."
        ),
        "Unexpected card charge": (
            "My credit card statement includes a purchase "
            "I did not authorize and my dispute is unresolved."
        ),
        "Mortgage payment problem": (
            "My mortgage servicer applied my monthly payment "
            "incorrectly and charged a late fee."
        ),
        "Bank account fee": (
            "My checking account was charged an overdraft fee "
            "although I had enough money in the account."
        ),
    }

    with input_col:
        choice = st.selectbox(
            "Try an example",
            list(examples),
            key="resolve_example_choice",
        )

        if st.session_state.get("resolve_previous_example") != choice:
            st.session_state["resolve_complaint"] = examples[choice]
            st.session_state["resolve_previous_example"] = choice
            st.session_state.pop("resolve_result", None)

        with st.form("resolve_classification_form"):
            complaint = st.text_area(
                "Complaint text",
                key="resolve_complaint",
                height=210,
                max_chars=10000,
                placeholder="Describe what happened...",
            )

            st.caption(
                "Use fictional examples without personal information."
            )

            submitted = st.form_submit_button(
                "Analyse complaint →"
            )

        if submitted:
            text = " ".join(complaint.lower().split())
            st.session_state.pop("resolve_result", None)

            if len(text.split()) < 5:
                st.warning(
                    "Enter at least five words describing the complaint."
                )
            else:
                features = model.named_steps["tfidf"].transform([text])

                if features.nnz == 0:
                    st.warning(
                        "Please enter a clearer financial complaint."
                    )
                else:
                    probabilities = model.predict_proba([text])[0]
                    predicted = model.predict([text])[0]

                    st.session_state["resolve_result"] = {
                        "label": predicted,
                        "scores": probabilities.tolist(),
                    }

        with st.expander("Supported categories"):
            for label in classes:
                st.write("• " + NAMES.get(label, label))

    with result_col:
        result = st.session_state.get("resolve_result")

        if result:
            category = html.escape(
                NAMES.get(result["label"], result["label"])
            )

            st.markdown(
                f"""
<div class="result-card">
    <small>SUGGESTED CATEGORY</small>
    <h3>{category}</h3>
</div>
""",
                unsafe_allow_html=True,
            )

            scores = pd.DataFrame({
                "Category": [
                    NAMES.get(label, label)
                    for label in classes
                ],
                "Score": result["scores"],
            }).sort_values("Score")

            fig = px.bar(
                scores,
                x="Score",
                y="Category",
                orientation="h",
                text="Score",
                color_discrete_sequence=["#087E8B"],
                title="Category scores",
            )

            fig.update_traces(
                texttemplate="%{x:.1%}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            )

            fig.update_xaxes(
                range=[0, 1.18],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                tickformat=".0%",
                title=None,
            )

            fig.update_yaxes(
                title=None,
                categoryorder="total ascending",
                tickfont=dict(size=12),
            )

            show_chart(
                fig,
                left=170,
                height=350,
            )

            st.caption(
                "Scores are model estimates, not verified confidence "
                "levels. Results reflect the last submitted complaint."
            )

        else:
            st.markdown(
                """
<div class="empty-result">
    <h3>Your analysis appears here</h3>
    <p>Write a complaint or select an example, then click
    <strong>Analyse complaint</strong>.</p>
    <p>You’ll see the suggested category and a comparison
    of all five category scores.</p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.caption(
            "The model always chooses one of five categories, even "
            "for unrelated text. Review suggestions before routing complaints."
        )

with analytics:
    st.subheader("Explore the evaluation dataset")
    st.caption(
        "Category distribution within the held-out test set."
    )

    data = report.loc[classes].copy()
    data["Category"] = [
        NAMES.get(label, label)
        for label in classes
    ]

    fig = px.bar(
        data.sort_values("support"),
        x="support",
        y="Category",
        color="Category",
        color_discrete_map=color_map,
        orientation="h",
        text="support",
        title="Test complaints by category",
        labels={"support": "Complaints"},
    )

    fig.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        range=[0, float(data["support"].max()) * 1.2]
    )
    fig.update_yaxes(title=None)

    show_chart(fig, left=180)

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:
        fig = px.pie(
            data,
            names="Category",
            values="support",
            color="Category",
            color_discrete_map=color_map,
            hole=0.62,
            title="Share of test complaints",
        )

        fig.update_traces(
            textinfo="percent",
            textposition="inside",
            marker=dict(
                line=dict(color="white", width=3)
            ),
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0,
            )
        )

        show_chart(
            fig,
            bottom=110,
            height=430,
        )

    with right:
        st.markdown("#### Category breakdown")

        breakdown = data[["Category", "support"]].copy()
        breakdown.columns = [
            "Category",
            "Test complaints",
        ]
        breakdown["Test complaints"] = (
            breakdown["Test complaints"].astype(int)
        )

        st.dataframe(
            breakdown,
            hide_index=True,
        )

        st.info(
            "Credit reporting is the largest category. Macro F1 "
            "gives every category equal weight during evaluation."
        )

    st.caption(
        "This dataset has no date column, so these charts show "
        "category composition rather than trends over time."
    )

with performance:
    st.subheader("Measure the model’s performance")

    st.caption(
        f"Evaluated on {metrics['testing_rows']:,} unseen complaints "
        "using an 80/20 stratified random split."
    )

    display = report.loc[
        classes,
        ["precision", "recall", "f1-score", "support"],
    ].copy()

    display.index = [
        NAMES.get(label, label)
        for label in classes
    ]

    display.columns = [
        "Precision",
        "Recall",
        "F1 Score",
        "Test Complaints",
    ]

    display["Test Complaints"] = (
        display["Test Complaints"].astype(int)
    )

    st.dataframe(
        display.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "Test Complaints": "{:,}",
        })
    )

    wrapped = {
        "credit_card": "Credit<br>Card",
        "credit_reporting": "Credit<br>Reporting",
        "debt_collection": "Debt<br>Collection",
        "mortgages_and_loans": "Mortgages<br>and Loans",
        "retail_banking": "Retail<br>Banking",
    }

    fig = px.imshow(
        matrix.to_numpy(),
        x=[
            wrapped.get(label, label)
            for label in matrix.columns
        ],
        y=[
            wrapped.get(label, label)
            for label in matrix.index
        ],
        text_auto=True,
        color_continuous_scale=[
            "#EDF6FA",
            "#8DCDD0",
            "#075A67",
        ],
        labels={
            "x": "Predicted category",
            "y": "Actual category",
            "color": "Complaints",
        },
        title="Confusion matrix",
        aspect="auto",
    )

    fig.update_xaxes(
        tickangle=0,
        title_standoff=20,
    )
    fig.update_yaxes(
        tickangle=0,
        title_standoff=20,
    )

    show_chart(
        fig,
        left=145,
        bottom=115,
        height=540,
    )

    comparison = pd.DataFrame({
        "Approach": [
            "Majority baseline",
            "Logistic Regression",
        ],
        "Accuracy": [
            metrics["baseline_accuracy"],
            metrics["accuracy"],
        ],
    })

    fig = px.bar(
        comparison,
        x="Approach",
        y="Accuracy",
        color="Approach",
        color_discrete_sequence=[
            "#BCD3DD",
            "#087E8B",
        ],
        text="Accuracy",
        title="Accuracy against the baseline",
    )

    fig.update_traces(
        texttemplate="%{y:.2%}",
        textposition="outside",
    )
    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
    )
    fig.update_xaxes(title=None)
    fig.update_layout(showlegend=False)

    show_chart(fig)

    st.download_button(
        "Download evaluation report",
        data=display.to_csv().encode("utf-8"),
        file_name="resolveai_evaluation.csv",
        mime="text/csv",
    )

with about:
    st.subheader("About ResolveAI")

    left, right = st.columns(
        [1, 1],
        gap="large",
    )

    with left:
        st.markdown(
            """
#### Purpose
Suggest a financial product category for a written consumer
complaint, helping support teams organise incoming messages.

#### Method
- TF-IDF text features.
- Logistic Regression with balanced class weights.
- Vocabulary learned from training complaints only.
- Evaluation using accuracy, Macro F1, and per-category metrics.

#### Data preparation
Removed missing text, normalised case and whitespace,
removed conflicting labels, and removed duplicate complaint text.
"""
        )

    with right:
        total_rows = (
            metrics["training_rows"] + metrics["testing_rows"]
        )

        st.markdown(
            f"""
#### Dataset
Consumer Complaints Dataset for NLP, downloaded from Kaggle.

The cleaned dataset contains **{total_rows:,} complaints**
across **{len(classes)} financial product categories**.

#### Limitations
- Training text was already preprocessed.
- Real user wording may differ.
- Similar templates may remain after exact duplicate removal.
- Random-split results do not establish future performance.
- The model does not assess whether a complaint is valid.
"""
        )

    st.link_button(
        "View dataset source",
        "https://www.kaggle.com/datasets/shashwatwork/"
        "consume-complaints-dataset-fo-nlp",
    )

st.markdown(
    """
<div class="resolve-footer">
    ResolveAI · Built by Anuvarna R. · Data Science & AI Portfolio
</div>
""",
    unsafe_allow_html=True,
)