import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
 
st.set_page_config(page_title="Walmart M5 Sales Forecasting Dashboard", layout="wide")
 
st.title("Walmart M5 Sales Forecasting Dashboard")
st.write("DEPI Graduation Project - Sales Analysis and Prediction")
 
# ------------------ PATHS ------------------
# All paths are relative to this script's location, so the app runs the
# same way regardless of whose machine it's deployed on.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.py may live either directly in the project root, or in its own
# subfolder (e.g. "app/") one level below the root where data/ and the
# model file actually are. Check both so this works either way.
CANDIDATE_ROOTS = [BASE_DIR, os.path.dirname(BASE_DIR)]
 
 
def find_existing_dir(*rel_parts):
    for root in CANDIDATE_ROOTS:
        candidate = os.path.join(root, *rel_parts)
        if os.path.isdir(candidate):
            return candidate
    # nothing found — return the first candidate so the error message below
    # at least shows a sensible expected path
    return os.path.join(CANDIDATE_ROOTS[0], *rel_parts)
 
 
def find_existing_file(path_options):
    """path_options: list of tuples, each a path (relative to a root) to try."""
    for root in CANDIDATE_ROOTS:
        for rel_parts in path_options:
            candidate = os.path.join(root, *rel_parts)
            if os.path.isfile(candidate):
                return candidate
    return os.path.join(CANDIDATE_ROOTS[0], *path_options[0])
 
 
DATA_DIR = find_existing_dir("data", "analysis")
MODEL_PATH = find_existing_file([
    ("models", "random_forest_model.pkl"),
    ("random_forest_model.pkl",),
])
 
EXPECTED_FILES = {
    "category_sales": os.path.join(DATA_DIR, "category_sales.parquet"),
    "department_sales": os.path.join(DATA_DIR, "department_sales.parquet"),
    "monthly_sales": os.path.join(DATA_DIR, "monthly_sales.parquet"),
    "state_sales": os.path.join(DATA_DIR, "state_sales.parquet"),
    "store_sales": os.path.join(DATA_DIR, "store_sales.parquet"),
    "weekday_sales": os.path.join(DATA_DIR, "weekday_sales.parquet"),
    "engineered_time_series": os.path.join(DATA_DIR, "engineered_time_series.csv"),
    "model": MODEL_PATH,
}
 
 
@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)
 
 
@st.cache_data
def load_engineered_series(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)
 
 
@st.cache_resource
def load_model(path):
    return joblib.load(path)
 
 
# ------------------ LOAD DATA (with a clear error if something is missing) ------------------
missing = [name for name, path in EXPECTED_FILES.items() if not os.path.exists(path)]
if missing:
    st.error(
        "Missing required file(s): "
        + ", ".join(missing)
        + f".\n\nExpected layout, relative to this script:\n"
        f"  {DATA_DIR}/category_sales.parquet\n"
        f"  {DATA_DIR}/department_sales.parquet\n"
        f"  {DATA_DIR}/monthly_sales.parquet\n"
        f"  {DATA_DIR}/state_sales.parquet\n"
        f"  {DATA_DIR}/store_sales.parquet\n"
        f"  {DATA_DIR}/weekday_sales.parquet\n"
        f"  {DATA_DIR}/engineered_time_series.csv\n"
        f"  {MODEL_PATH}"
    )
    st.stop()
 
category_sales = load_parquet(EXPECTED_FILES["category_sales"])
department_sales = load_parquet(EXPECTED_FILES["department_sales"])
monthly_sales = load_parquet(EXPECTED_FILES["monthly_sales"])
state_sales = load_parquet(EXPECTED_FILES["state_sales"])
store_sales = load_parquet(EXPECTED_FILES["store_sales"])
weekday_sales = load_parquet(EXPECTED_FILES["weekday_sales"])
ts_df = load_engineered_series(EXPECTED_FILES["engineered_time_series"])
model = load_model(EXPECTED_FILES["model"])
 
MODEL_FEATURES = list(model.feature_names_in_)
 
# ------------------ SIDEBAR MENU ------------------
page = st.sidebar.radio("Choose a page", ["EDA", "Prediction"])
 
# ------------------ PAGE 1: EDA ------------------
if page == "EDA":
 
    st.header("Exploratory Data Analysis")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", int(monthly_sales["total_sales"].sum()))
    col2.metric("Categories", category_sales["cat_id"].nunique())
    col3.metric("Stores", store_sales["store_id"].nunique())
 
    st.subheader("Monthly Total Sales Trend")
    monthly_sorted = monthly_sales.copy()
    monthly_sorted["year_month"] = pd.to_datetime(
        monthly_sorted["year"].astype(str) + "-" + monthly_sorted["month"].astype(str) + "-01"
    )
    monthly_sorted = monthly_sorted.sort_values("year_month")
    fig1 = px.line(monthly_sorted, x="year_month", y="total_sales")
    fig1.update_layout(xaxis_title="Month", yaxis_title="Total Sales")
    st.plotly_chart(fig1, use_container_width=True)
 
    st.subheader("Total Sales by Category")
    fig2 = px.bar(
        category_sales.sort_values("total_sales", ascending=False),
        x="cat_id", y="total_sales",
    )
    st.plotly_chart(fig2, use_container_width=True)
 
    col4, col5 = st.columns(2)
 
    with col4:
        st.subheader("Total Sales by Store")
        fig3 = px.bar(
            store_sales.sort_values("total_sales", ascending=False),
            x="store_id", y="total_sales", color="state_id",
        )
        st.plotly_chart(fig3, use_container_width=True)
 
    with col5:
        st.subheader("Total Sales by State")
        fig4 = px.pie(state_sales, names="state_id", values="total_sales")
        st.plotly_chart(fig4, use_container_width=True)
 
    st.subheader("Total Sales by Weekday")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_sorted = weekday_sales.set_index("weekday").reindex(weekday_order).reset_index()
    fig5 = px.bar(weekday_sorted, x="weekday", y="total_sales")
    st.plotly_chart(fig5, use_container_width=True)
 
    with st.expander("Total Sales by Department"):
        fig6 = px.bar(
            department_sales.sort_values("total_sales", ascending=False),
            x="dept_id", y="total_sales", color="cat_id",
        )
        st.plotly_chart(fig6, use_container_width=True)
 
# ------------------ PAGE 2: PREDICTION ------------------
if page == "Prediction":
 
    st.header("Predict Daily Sales")
    st.write(
        "Pick a date from the historical dataset. The model's actual input features "
        "for that day — including sales lags and rolling averages — are pulled directly "
        "from the data, so the prediction reflects real conditions rather than placeholder "
        "zeros. You can optionally override the day of week or SNAP status to explore "
        "what-if scenarios."
    )
 
    min_date = ts_df["date"].min().date()
    max_date = ts_df["date"].max().date()
 
    selected_date = st.date_input(
        "Select a date", value=max_date, min_value=min_date, max_value=max_date
    )
 
    row = ts_df[ts_df["date"] == pd.to_datetime(selected_date)]
 
    if row.empty:
        st.warning("No data available for this date.")
    else:
        base_row = row.iloc[0]
        actual_sales = base_row["sales"]
 
        st.subheader("What-if adjustments (optional)")
        col1, col2 = st.columns(2)
 
        wday_map = {
            "Saturday": 1, "Sunday": 2, "Monday": 3, "Tuesday": 4,
            "Wednesday": 5, "Thursday": 6, "Friday": 7,
        }
        # weekday_Friday was dropped as the baseline category during one-hot encoding,
        # so "Friday" is represented by all weekday_* dummies being 0.
        weekday_dummy_cols = [
            "weekday_Monday", "weekday_Saturday", "weekday_Sunday",
            "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday",
        ]
 
        with col1:
            override_weekday = st.checkbox("Override day of week")
            day_choice = None
            if override_weekday:
                day_choice = st.selectbox("Day of Week", list(wday_map.keys()))
 
        with col2:
            override_snap = st.checkbox("Override SNAP CA status")
            snap_choice = None
            if override_snap:
                snap_choice = st.selectbox("SNAP CA", ["No", "Yes"])
 
        # Start from the real feature row for the selected date.
        input_df = pd.DataFrame([base_row[MODEL_FEATURES].values], columns=MODEL_FEATURES)
 
        if override_weekday and day_choice:
            input_df["wday"] = wday_map[day_choice]
            input_df["is_weekend"] = 1 if day_choice in ["Saturday", "Sunday"] else 0
            for c in weekday_dummy_cols:
                if c in input_df.columns:
                    input_df[c] = 0
            dummy_col = f"weekday_{day_choice}"
            if dummy_col in input_df.columns:
                input_df[dummy_col] = 1
 
        if override_snap and snap_choice:
            if "snap_CA" in input_df.columns:
                input_df["snap_CA"] = 1 if snap_choice == "Yes" else 0
 
        input_df = input_df.astype(float)
 
        prediction = model.predict(input_df)[0]
 
        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Sales (historical)", int(actual_sales))
        c2.metric("Model Prediction", round(float(prediction), 1))
        c3.metric("Difference", round(float(prediction) - float(actual_sales), 1))
 
        with st.expander("See exact feature values sent to the model"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))