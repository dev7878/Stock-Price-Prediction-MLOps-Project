import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Test App - Streamlit Cloud Deployment")
st.write("If you can see this, Streamlit Cloud is working!")

st.success("✅ Deployment successful!")
st.info("This is a minimal test app to verify deployment.")

# Add a simple interactive element
name = st.text_input("Enter your name:", "MLOps Engineer")
st.write(f"Hello, {name}! Your app is working perfectly.")

# Add a chart to test plotly
import plotly.express as px
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = np.random.randn(30).cumsum() + 100

df = pd.DataFrame({
    'Date': dates,
    'Price': prices
})

fig = px.line(df, x='Date', y='Price', title='Sample Stock Price Chart')
st.plotly_chart(fig, use_container_width=True)

st.balloons()
st.write("🎉 **Your Streamlit Cloud deployment is working!**")
