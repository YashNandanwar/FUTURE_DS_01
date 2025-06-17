import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Social Media Dashboard", layout="wide")

# Title
st.title("Social Media Analytics Dashboard")

# Load CSV data
@st.cache_data
def load_data():
    return pd.read_csv("sentimentdataset.csv")

df = load_data()

# Convert timestamp column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Display raw data
st.subheader("Raw Data")
st.dataframe(df)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Posts", len(df))
col2.metric("Avg Likes", round(df['Likes'].mean(), 2))
col3.metric("Avg Retweets", round(df['Retweets'].mean(), 2))

# Visualizations
st.subheader("Data Visualizations")

# Sentiment Distribution
fig1 = px.pie(df, names='Sentiment', title='Sentiment Distribution')
st.plotly_chart(fig1, use_container_width=True)

# Posts by Platform
fig2 = px.histogram(df, x='Platform', title='Posts by Platform')
st.plotly_chart(fig2, use_container_width=True)

# Posts over time
posts_over_time = df.groupby('Timestamp').size().reset_index(name='Count')
fig3 = px.line(posts_over_time, x='Timestamp', y='Count', title='Posts Over Time')
st.plotly_chart(fig3, use_container_width=True)

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect(
    "Select Country", df['Country'].unique(), default=df['Country'].unique()
)
selected_platform = st.sidebar.multiselect(
    "Select Platform", df['Platform'].unique(), default=df['Platform'].unique()
)

# Apply filters
filtered_df = df[df['Country'].isin(selected_country) & df['Platform'].isin(selected_platform)]

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)
