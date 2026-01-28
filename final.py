import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="YouTube Ad Revenue Analysis",
    layout="wide"
)

st.title("📊 YouTube Ad Revenue Analysis & Prediction")

# --------------------------------------------------
# Load data & trained pipeline
# --------------------------------------------------
df = pd.read_csv("youtube_ad_revenue_dataset.csv")

with open("best_model_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "EDA & Insights", "Revenue Prediction", "Business Insights"]
)

# --------------------------------------------------
# OVERVIEW SECTION
# --------------------------------------------------
if menu == "Overview":
    st.subheader("📌 Project Overview")

    st.markdown("""
    This project analyzes **YouTube video performance data** to understand
    the factors affecting **ad revenue** and uses a machine learning pipeline
    to predict future revenue.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Videos", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Target Variable", "ad_revenue_usd")

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

# --------------------------------------------------
# EDA & INSIGHTS SECTION
# --------------------------------------------------
elif menu == "EDA & Insights":
    st.subheader("📈 Exploratory Data Analysis")

    st.markdown("### Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("""
    🔍 **Key Observations**
    - Watch time and views strongly influence ad revenue
    - Engagement metrics have moderate impact
    - Video length alone does not guarantee higher revenue
    """)

    st.markdown("### Average Ad Revenue by Category")

    fig2, ax2 = plt.subplots()
    df.groupby("category")["ad_revenue_usd"].mean().sort_values().plot(
        kind="barh", ax=ax2
    )
    ax2.set_xlabel("Average Ad Revenue (USD)")
    st.pyplot(fig2)

# --------------------------------------------------
# REVENUE PREDICTION SECTION
# --------------------------------------------------
elif menu == "Revenue Prediction":
    st.subheader("🤖 Ad Revenue Prediction")

    st.markdown("Enter video performance details to predict ad revenue.")

    col1, col2, col3 = st.columns(3)
    views = col1.number_input("Views", min_value=0)
    likes = col2.number_input("Likes", min_value=0)
    comments = col3.number_input("Comments", min_value=0)

    col4, col5, col6 = st.columns(3)
    watch_time = col4.number_input("Watch Time (minutes)", min_value=0.0)
    video_length = col5.number_input("Video Length (minutes)", min_value=0.0)
    subscribers = col6.number_input("Subscribers", min_value=0)

    col7, col8, col9 = st.columns(3)
    category = col7.selectbox("Category", df["category"].unique())
    device = col8.selectbox("Device", df["device"].unique())
    country = col9.selectbox("Country", df["country"].unique())

    if st.button("Predict Ad Revenue"):
        input_df = pd.DataFrame({
            "views": [views],
            "likes": [likes],
            "comments": [comments],
            "watch_time_minutes": [watch_time],
            "video_length_minutes": [video_length],
            "subscribers": [subscribers],
            "category": [category],
            "device": [device],
            "country": [country]
        })

        prediction = model.predict(input_df)

        st.success(f"💰 Predicted Ad Revenue: ${prediction[0]:,.2f}")

# --------------------------------------------------
# BUSINESS INSIGHTS SECTION
# --------------------------------------------------
elif menu == "Business Insights":
    st.subheader("📌 Business Insights & Recommendations")

    st.markdown("""
    ### 📊 Key Insights
    ✅ Watch time and views are the strongest revenue drivers  
    ✅ Engagement metrics have secondary impact  
    ✅ Device and country affect monetization rates  
    ✅ Linear and tree-based models perform competitively  

    ### 🎯 Recommendations
    🔹 Focus on increasing watch time and retention  
    🔹 Optimize content for high-performing devices  
    🔹 Target high-revenue regions  
    🔹 Use ML predictions for content planning and forecasting
    """)
