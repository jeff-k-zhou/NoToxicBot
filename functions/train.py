import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

def extractData(filename):
    data = pd.read_csv(filename, encoding="utf8")
    
    return data

def preprocess(string):
    data = pd.read_csv('toxicity_en.csv', encoding="utf8")

    enc = LabelEncoder()
    label = enc.fit_transform(data["is_toxic"])

    text = data["text"]

    processed = text.str.lower()
    #processed = processed.str.replace(r'\W+\S+', '', regex=True)
    processed = processed.str.replace(r'^[^ ]+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
    processed = processed.str.replace(r'£|\$', '')
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenum')
    processed = processed.str.replace(r'\d+(\.\d+)?', '')
    processed = processed.str.replace(r'can\'t', 'cannot')
    processed = processed.str.replace(r'won\'t', 'willnot')
    processed = processed.str.replace(r'n\'t', 'not')
    processed = processed.str.replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    #processed = processed.str.replace(r'[^\w\s]', ' ')
    #processed = processed.str.replace(r'\s+', ' ')
    processed = processed.str.strip()

    stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    ps = nltk.PorterStemmer()
    processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

    words = []
    for msg in processed:
        words.extend(nltk.word_tokenize(msg))

    words = nltk.FreqDist(words)
    word_features = [x[0] for x in words.most_common(1500)]
    return processed, label, word_features

def find_features(message, word_features):
    words = nltk.word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

def train():
    processed, label, word_features = preprocess() 
    messages = list(zip(processed, label))

    names = ['K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SGD Classifier',
            'Naive Bayes', 'Support Vector Classifier']

    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier(max_iter=100),
        MultinomialNB(),
        SVC(kernel='linear')
    ]

    models = list(zip(names, classifiers))
    nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard', n_jobs=-1))

    np.random.shuffle(messages)
    feature_set = [(find_features(text, word_features), label) for (text, label) in messages]
    training, test = train_test_split(feature_set, test_size=0.2, random_state=1)
    nltk_ensemble.train(training)
    t_accuracy = nltk.classify.accuracy(nltk_ensemble, training)
    accuracy = nltk.classify.accuracy(nltk_ensemble, test)

    text_features, labels = zip(*test)
    prediction = nltk_ensemble.classify_many(text_features)
    print(prediction[:10])
    return nltk_ensemble

def predict(text, model=SklearnClassifier):
    features = find_features(text)
    return model.classify(features)

model = train()
print(predict("I am a good person", model))



