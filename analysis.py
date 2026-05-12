# =========================================
# GOOGLE PLAY STORE DATA ANALYTICS PROJECT
# =========================================

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =========================================
# LOAD DATASETS
# =========================================

apps = pd.read_csv("data/apps.csv")
reviews = pd.read_csv("data/user_reviews.csv")

# =========================================
# BASIC INFO
# =========================================

print("\nFIRST 5 ROWS")
print(apps.head())

print("\nDATASET INFO")
print(apps.info())

print("\nMISSING VALUES")
print(apps.isnull().sum())

# =========================================
# DATA CLEANING
# =========================================

# REMOVE DUPLICATES
apps.drop_duplicates(inplace=True)

# REMOVE NULL RATINGS
apps.dropna(subset=['Rating'], inplace=True)

# CLEAN INSTALLS COLUMN
apps['Installs'] = apps['Installs'].str.replace('+', '', regex=False)
apps['Installs'] = apps['Installs'].str.replace(',', '', regex=False)
apps['Installs'] = apps['Installs'].astype(int)

# CLEAN PRICE COLUMN
apps['Price'] = apps['Price'].str.replace('$', '', regex=False)
apps['Price'] = apps['Price'].astype(float)

# CLEAN REVIEWS COLUMN
apps['Reviews'] = pd.to_numeric(apps['Reviews'])

# CLEAN SIZE COLUMN
def convert_size(size):

    if 'M' in str(size):
        return float(size.replace('M', ''))

    elif 'k' in str(size):
        return float(size.replace('k', '')) / 1024

    else:
        return np.nan

apps['Size'] = apps['Size'].apply(convert_size)

# =========================================
# EDA VISUALIZATIONS
# =========================================

sns.set_style("whitegrid")

# -----------------------------------------
# TOP CATEGORIES
# -----------------------------------------

plt.figure(figsize=(12, 6))

apps['Category'].value_counts().head(10).plot(
    kind='bar'
)

plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------------------
# RATINGS DISTRIBUTION
# -----------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    apps['Rating'],
    bins=20,
    kde=True
)

plt.title("Ratings Distribution")
plt.xlabel("Rating")

plt.show()

# -----------------------------------------
# FREE VS PAID APPS
# -----------------------------------------

plt.figure(figsize=(6, 6))

apps['Type'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Free vs Paid Apps")
plt.ylabel("")

plt.show()

# -----------------------------------------
# TOP INSTALLED CATEGORIES
# -----------------------------------------

category_installs = apps.groupby(
    'Category'
)['Installs'].sum()

plt.figure(figsize=(12, 6))

category_installs.sort_values(
    ascending=False
).head(10).plot(kind='bar')

plt.title("Top Installed Categories")
plt.xlabel("Category")
plt.ylabel("Total Installs")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------------------
# PRICE VS RATING
# -----------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x='Price',
    y='Rating',
    data=apps
)

plt.title("Price vs Rating")

plt.show()

# -----------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------

numeric_df = apps[
    ['Rating', 'Reviews', 'Installs', 'Price']
]

corr = numeric_df.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# =========================================
# SENTIMENT ANALYSIS
# =========================================

reviews.dropna(
    subset=['Translated_Review'],
    inplace=True
)

# SENTIMENT FUNCTION
def get_sentiment(review):

    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"

reviews['Sentiment'] = reviews[
    'Translated_Review'
].apply(get_sentiment)

print("\nSENTIMENT SAMPLE")
print(
    reviews[
        ['Translated_Review', 'Sentiment']
    ].head()
)

# -----------------------------------------
# SENTIMENT CHART
# -----------------------------------------

plt.figure(figsize=(8, 5))

reviews['Sentiment'].value_counts().plot(
    kind='bar'
)

plt.title("User Review Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()

# =========================================
# WORD CLOUD
# =========================================

text = ' '.join(
    reviews['Translated_Review'].astype(str)
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(text)

plt.figure(figsize=(15, 7))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Most Common Words in Reviews")

plt.show()

# =========================================
# TOP RATED APPS
# =========================================

top_apps = apps.sort_values(
    by='Rating',
    ascending=False
)

print("\nTOP RATED APPS")

print(
    top_apps[
        ['App', 'Category', 'Rating']
    ].head(10)
)

# =========================================
# MACHINE LEARNING MODEL
# =========================================

ml_data = apps[
    ['Reviews', 'Installs', 'Price', 'Rating']
].dropna()

X = ml_data[
    ['Reviews', 'Installs', 'Price']
]

y = ml_data['Rating']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = LinearRegression()

model.fit(X_train, y_train)

# ACCURACY
score = model.score(X_test, y_test)

print("\nMODEL ACCURACY")
print(score)

# SAMPLE PREDICTION
prediction = model.predict(
    [[50000, 1000000, 0]]
)

print("\nPREDICTED RATING")
print(prediction[0])

# =========================================
# FINAL INSIGHTS
# =========================================

print("\nKEY INSIGHTS")

print("1. Most apps on Play Store are free.")
print("2. Family and Game categories dominate.")
print("3. Higher installs generally relate to better ratings.")
print("4. Most user sentiments are positive.")
print("5. Paid apps are fewer compared to free apps.")
