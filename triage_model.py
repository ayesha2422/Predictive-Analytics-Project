from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_triage_model():
    symptoms = [
        "severe chest pain",
        "minor headache",
        "heavy bleeding",
        "broken arm pain",
        "mild fever"
    ]

    urgency = [3, 1, 3, 2, 1]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptoms)

    model = MultinomialNB()
    model.fit(X, urgency)

    return model, vectorizer
