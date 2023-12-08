# MachineLearningModule.py
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

class MachineLearningModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        # feed forward neural network for classification, 1 hidden layer of 5 neurons, 'ReLU' activation function, 'Adam' optimization algorithm
        self.model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=300, activation='relu', solver='adam', random_state=1)
        self.is_trained = False

    def train(self):
        # read relevant phrases
        relevant_phrases = pd.read_csv('relevant_phrases.csv', header=None)
        # convert first column of dataframe to a list
        relevant_phrases = relevant_phrases.iloc[:,0].tolist()
        
        # using the saved csv data from previous runs from early november 2023 as trainig data
        df = pd.read_csv('job_listings.csv')
        # Create a new column 'label' that stores the information if the job title is contained in any of the relevant phrases (1) or not (0)
        df['label'] = df['job_title'].apply(lambda title: is_relevant_job_title(title, relevant_phrases))
        
        # Preprocess the data (# Adapted from https://www.geeksforgeeks.org/what-is-the-difference-between-transform-and-fit_transform-in-sklearn-python/, last accessed: Nov 2023)
        # Independent variable (X): 'job_title' holds the title of the job
        x = self.vectorizer.fit_transform(df['job_title']).toarray()

        # Dependent variable (Y): 'label' (1 if job_title is relevant, otherwise 0)
        y = df['label']

        # Train and evaluate the model using cross-validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        scores = cross_val_score(self.model, x, y, cv=skf)

        # mean of the cross-validation scores
        print(f"Mean accuracy: {scores.mean()}")
        
        # Training the model on the entire dataset
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, new_data):
        if not self.is_trained:
            raise Exception("Model must be trained before prediction.")

        # Preprocess the new job descriptions
        df = new_data['job_title']
        new_X = self.vectorizer.transform(df).toarray()
        
        # Use the trained model to predict relevance
        return self.model.predict(new_X)

def is_relevant_job_title(job_title, relevant_phrases):
    normalized_title = job_title.lower().replace('-', ' ').replace('/', ' ')
    for phrase in relevant_phrases:
        normalized_phrase = phrase.lower().replace('-', ' ').replace('/', ' ')
        
        if re.search(r'\b' + re.escape(normalized_phrase) + r'\b', normalized_title):
            return True
    return False
