from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the Dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
data = fetch_20newsgroups(categories=categories)

# Split the data into test/training sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Build Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print(metrics.classification_report(y_test, y_pred, target_names=data.target_names))
