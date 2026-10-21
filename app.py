
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Upload a bank customer CSV file to train a model, "
    "estimate churn probabilities, and classify customer risk."
)

st.info(
    "Your CSV must contain an Exited column: "
    "1 means the customer left; 0 means the customer stayed."
)

uploaded_file = st.file_uploader(
    "Upload customer data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        if "Exited" not in df.columns:
            st.error("The CSV must contain an 'Exited' column.")
        elif df.empty or df["Exited"].nunique() < 2:
            st.error(
                "Please upload data containing both churn classes: "
                "Exited = 0 and Exited = 1."
            )
        else:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Train Model and Calculate Risk"):
                data = df.dropna(subset=["Exited"]).copy()
                y = pd.to_numeric(data.pop("Exited"), errors="coerce")
                valid = y.isin([0, 1])
                data = data.loc[valid]
                y = y.loc[valid].astype(int)

                if y.nunique() < 2:
                    st.error("The dataset must contain both 0 and 1 labels.")
                elif len(data) < 10:
                    st.error("Please upload at least 10 valid customer rows.")
                else:
                    # Remove identifiers that should not drive predictions.
                    data = data.drop(
                        columns=["CustomerId", "Surname"],
                        errors="ignore"
                    )

                    numeric_cols = data.select_dtypes(
                        include=["number", "bool"]
                    ).columns.tolist()
                    categorical_cols = [
                        c for c in data.columns
                        if c not in numeric_cols
                    ]

                    transformers = []

                    if numeric_cols:
                        transformers.append((
                            "numeric",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="median"))
                            ]),
                            numeric_cols
                        ))

                    if categorical_cols:
                        transformers.append((
                            "categorical",
                            Pipeline([
                                ("imputer", SimpleImputer(
                                    strategy="most_frequent"
                                )),
                                ("encoder", OneHotEncoder(
                                    handle_unknown="ignore"
                                ))
                            ]),
                            categorical_cols
                        ))

                    if not transformers:
                        st.error("No usable customer features were found.")
                    else:
                        preprocessor = ColumnTransformer(transformers)

                        model = Pipeline([
                            ("preprocessor", preprocessor),
                            ("classifier", RandomForestClassifier(
                                n_estimators=100,
                                random_state=42,
                                class_weight="balanced"
                            ))
                        ])

                        stratify = y if y.value_counts().min() >= 2 else None

                        X_train, X_test, y_train, y_test = train_test_split(
                            data,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=stratify
                        )

                        model.fit(X_train, y_train)
                        probabilities = model.predict_proba(data)[
                            :, list(model.named_steps["classifier"].classes_).index(1)
                        ]

                        results = df.loc[data.index].copy()
                        results["Churn Probability (%)"] = (
                            probabilities * 100
                        ).round(2)
                        results["Risk Category"] = pd.cut(
                            probabilities,
                            bins=[-0.001, 0.30, 0.60, 1.001],
                            labels=["Low", "Medium", "High"]
                        )

                        st.subheader("Model Results")
                        st.metric(
                            "Test Accuracy",
                            f"{accuracy_score(y_test, model.predict(X_test)):.1%}"
                        )

                        counts = results["Risk Category"].value_counts()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Low Risk", int(counts.get("Low", 0)))
                        c2.metric("Medium Risk", int(counts.get("Medium", 0)))
                        c3.metric("High Risk", int(counts.get("High", 0)))

                        st.dataframe(results, use_container_width=True)

                        st.download_button(
                            "Download Risk Results CSV",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="churn_risk_results.csv",
                            mime="text/csv"
                        )

    except Exception as e:
        st.error(f"Could not process the file: {e}")
