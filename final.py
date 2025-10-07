st.write("Customer-Churn-Dashboard!")
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

df = pd.read_csv("customer_summary.csv")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title(" Customer Churn Dashboard")
st.markdown("**By Hussein Mohamed Yousry | Data Science Final Project**")

st.sidebar.header(" Key Metrics & Filters")

gender_filter = st.sidebar.multiselect("Select Gender", options=df['gender'].dropna().unique())
wealth_filter = st.sidebar.multiselect("Select Wealth Segment", options=df['wealth_segment'].dropna().unique())
car_filter = st.sidebar.multiselect("Select Car Ownership", options=df['owns_car'].dropna().unique())

df_filtered = df.copy()
if gender_filter:
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]
if wealth_filter:
    df_filtered = df_filtered[df_filtered['wealth_segment'].isin(wealth_filter)]
if car_filter:
    df_filtered = df_filtered[df_filtered['owns_car'].isin(car_filter)]


total_customers = df_filtered['customer_id'].nunique()
churn_rate = round((df_filtered['churn'].mean()) * 100, 2)
avg_spent = round(df_filtered['total_spent'].mean(), 2) if "total_spent" in df_filtered else 0
avg_profit = round(df_filtered['total_profit'].mean(), 2) if "total_profit" in df_filtered else 0

st.sidebar.metric("Total Customers", total_customers)
st.sidebar.metric("Churn Rate (%)", churn_rate)
st.sidebar.metric("Avg Spending", f"${avg_spent}")
st.sidebar.metric("Avg Profit", f"${avg_profit}")

tab1, tab2, tab3 = st.tabs([" Customer Insights", " Transactions", " Churn Prediction"])


with tab1:
    st.header("Customer Overview")
    st.markdown("Explore customer demographics, wealth, and behavior.")

    if "age" in df_filtered.columns:
        fig_age = px.histogram(df_filtered, x="age", nbins=20, title="Customer Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    if "wealth_segment" in df_filtered.columns:
        fig_ws = px.histogram(df_filtered, x="wealth_segment", color="churn", barmode="group",
                              title="Wealth Segment vs Churn")
        st.plotly_chart(fig_ws, use_container_width=True)

    if "owns_car" in df_filtered.columns:
        fig_car = px.histogram(df_filtered, x="owns_car", color="churn", barmode="group",
                               title="Car Ownership vs Churn")
        st.plotly_chart(fig_car, use_container_width=True)

    if "job_industry_category" in df_filtered.columns:
        job_counts = df_filtered['job_industry_category'].value_counts().reset_index()
        job_counts.columns = ['job_industry_category', 'count']
        job_counts = job_counts.sort_values(by="count", ascending=False)

        fig_job = px.bar(job_counts,
                         x='job_industry_category',
                         y='count',
                         title="Job Industry Distribution")
        st.plotly_chart(fig_job, use_container_width=True)


with tab2:
    st.header("Transaction Insights")
    st.markdown("Analyze transaction patterns, spending, and churn impact.")

    if "product_line" in df_filtered.columns and "total_profit" in df_filtered.columns:
        fig_pl = px.bar(df_filtered, x="product_line", y="total_profit", color="churn",
                        title="Total Profit by Product Line", barmode="group")
        st.plotly_chart(fig_pl, use_container_width=True)

    if "profit_margin" in df_filtered.columns:
        fig_pm = px.histogram(df_filtered, x="profit_margin", nbins=30, color="churn",
                              title="Profit Margin Distribution")
        st.plotly_chart(fig_pm, use_container_width=True)

    if "online_order" in df_filtered.columns:
        fig_order = px.pie(df_filtered, names="online_order", title="Online vs In-Store Orders")
        st.plotly_chart(fig_order, use_container_width=True)

    if "total_spent" in df_filtered.columns:
        fig_churn_spent = px.box(df_filtered, x="churn", y="total_spent",
                                 title="Spending by Churn Status")
        st.plotly_chart(fig_churn_spent, use_container_width=True)

    numeric_cols = ['frequency', 'total_spent', 'total_profit', 'profit_margin', 'age', 'churn']
    available_cols = [col for col in numeric_cols if col in df_filtered.columns]

    if available_cols:
        corr = df_filtered[available_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)


with tab3:
    st.header("Churn Prediction")
    st.markdown("Upload customer data and predict churn using trained models.")

    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

    if model_choice == "Logistic Regression":
        model = joblib.load("logistic_regression_model1.pkl")
    elif model_choice == "Random Forest":
        model = joblib.load("random_forest_model2.pkl")
    else:
        model = joblib.load("gradient_boosting_model3.pkl")

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        predictions = model.predict(new_data)
        st.write("### Predictions")
        st.write(predictions)
