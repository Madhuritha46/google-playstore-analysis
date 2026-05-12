# ======================================
# GOOGLE PLAY STORE ANALYTICS DASHBOARD
# ======================================

# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Google Play Store Analytics",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# TITLE
# ======================================

st.title("📱 Google Play Store Analytics Dashboard")

st.markdown("---")

# ======================================
# LOAD DATA
# ======================================

apps = pd.read_csv("data/apps.csv")

# ======================================
# DATA CLEANING
# ======================================

apps.drop_duplicates(inplace=True)

apps.dropna(subset=['Rating'], inplace=True)

# CLEAN INSTALLS
apps['Installs'] = apps['Installs'].str.replace(
    '+',
    '',
    regex=False
)

apps['Installs'] = apps['Installs'].str.replace(
    ',',
    '',
    regex=False
)

apps['Installs'] = apps['Installs'].astype(int)

# CLEAN PRICE
apps['Price'] = apps['Price'].str.replace(
    '$',
    '',
    regex=False
)

apps['Price'] = apps['Price'].astype(float)

# CLEAN REVIEWS
apps['Reviews'] = pd.to_numeric(
    apps['Reviews']
)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.header("Dashboard Filters")

category = st.sidebar.selectbox(
    "Select Category",
    sorted(apps['Category'].unique())
)

rating_filter = st.sidebar.slider(
    "Minimum Rating",
    1.0,
    5.0,
    4.0
)

# FILTER DATA
filtered_data = apps[
    (apps['Category'] == category) &
    (apps['Rating'] >= rating_filter)
]

# ======================================
# SEARCH FEATURE
# ======================================

search = st.text_input(
    "🔍 Search App"
)

if search:

    filtered_data = filtered_data[
        filtered_data['App'].str.contains(
            search,
            case=False
        )
    ]

# ======================================
# KPI SECTION
# ======================================

st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Apps",
    len(filtered_data)
)

col2.metric(
    "Average Rating",
    round(filtered_data['Rating'].mean(), 2)
)

col3.metric(
    "Total Installs",
    f"{filtered_data['Installs'].sum():,}"
)

st.markdown("---")

# ======================================
# DATA TABLE
# ======================================

st.subheader("📋 App Data")

st.dataframe(
    filtered_data.head(20)
)

# ======================================
# RATING DISTRIBUTION
# ======================================

st.subheader("⭐ Ratings Distribution")

fig1 = px.histogram(
    filtered_data,
    x='Rating',
    nbins=20,
    title='Ratings Distribution'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ======================================
# INSTALLS VS RATING
# ======================================

st.subheader("📈 Installs vs Rating")

fig2 = px.scatter(
    filtered_data,
    x='Rating',
    y='Installs',
    color='Type',
    hover_data=['App'],
    title='Installs vs Rating'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ======================================
# FREE VS PAID
# ======================================

st.subheader("💰 Free vs Paid Apps")

fig3 = px.pie(
    filtered_data,
    names='Type',
    title='Free vs Paid Apps'
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ======================================
# TOP REVIEWED APPS
# ======================================

st.subheader("🏆 Top Reviewed Apps")

top_reviewed = filtered_data.sort_values(
    by='Reviews',
    ascending=False
).head(10)

fig4 = px.bar(
    top_reviewed,
    x='App',
    y='Reviews',
    title='Top Reviewed Apps'
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ======================================
# CORRELATION HEATMAP
# ======================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = apps[
    ['Rating', 'Reviews', 'Installs', 'Price']
]

corr = numeric_df.corr()

fig5 = px.imshow(
    corr,
    text_auto=True,
    title='Correlation Heatmap'
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# ======================================
# MACHINE LEARNING MODEL
# ======================================

ml_data = apps[
    ['Reviews', 'Installs', 'Price', 'Rating']
].dropna()

X = ml_data[
    ['Reviews', 'Installs', 'Price']
]

y = ml_data['Rating']

# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

# ======================================
# PREDICTION SECTION
# ======================================

st.subheader("🤖 Predict App Rating")

reviews_input = st.number_input(
    "Number of Reviews",
    value=5000
)

installs_input = st.number_input(
    "Number of Installs",
    value=100000
)

price_input = st.number_input(
    "Price",
    value=0.0
)

prediction = model.predict([
    [
        reviews_input,
        installs_input,
        price_input
    ]
])

st.success(
    f"Predicted Rating: {prediction[0]:.2f}"
)

# ======================================
# DOWNLOAD FEATURE
# ======================================

csv = filtered_data.to_csv(
    index=False
).encode('utf-8')

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name="filtered_apps.csv",
    mime="text/csv"
)

# ======================================
# TOP APPS TABLE
# ======================================

st.subheader("🌟 Top Rated Apps")

top_apps = filtered_data.sort_values(
    by='Rating',
    ascending=False
)

st.dataframe(
    top_apps[
        ['App', 'Category', 'Rating', 'Installs']
    ].head(10)
)

# ======================================
# FOOTER
# ======================================

st.markdown("---")


