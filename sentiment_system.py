import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("comments.csv")

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

print(predict_sentiment("The teacher is kind but homework is too heavy"))

positive_text = " ".join(df[df["label"]==1]["text"])
wc = WordCloud(width=800, height=400, background_color="white")
wc.generate(positive_text)

plt.imshow(wc)
plt.axis("off")
plt.show()
