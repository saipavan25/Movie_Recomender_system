from flask import Flask, request, render_template, jsonify, url_for
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

app = Flask(__name__)



movies = pd.read_csv("movies.csv")
cv=TfidfVectorizer()
transformer=cv.fit(movies['genres'])
indices=pd.Series(movies.index,index=movies['title'])
titles=movies['title']

tfidf_matrix=transformer.transform(movies['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# TODO: add versioning to url
@app.route('/', methods=['GET', 'POST'])
def predict():
    """ Main webpage with user input through form and prediction displayed

    :return: main webpage host, displays prediction if user submitted in text field
    """

    if request.method == 'POST':

        # response = request.form['text']
        input_text = request.form['movie']
        recommendation = recommendations(input_text)
        recommendation=recommendation.tolist()
        print(recommendation)
        return render_template('index.html', submission=recommendation)

    if request.method == 'GET':
        return render_template('index.html')


def recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

if __name__ == '__main__':
    app.run(debug=True)
