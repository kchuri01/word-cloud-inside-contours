# Author: Kajal Churi
# Project: Word Cloud Inside Contours

# pip install wordcloud pillow matplotlib scikit-learn requests

import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Download a sample text (The Adventures of Sherlock Holmes)
url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = requests.get(url)
book_text = response.text

# Compute TF-IDF scores for words
corpus = [book_text]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
tfidf_scores = tfidf_matrix.toarray()[0]
words = vectorizer.get_feature_names_out()
word_tfidf = dict(zip(words, tfidf_scores))

# Load the contour mask (update path to your local image)
mask = np.array(Image.open("plane.jpeg"))

# Generate
wc = WordCloud(
    background_color="white",
    mask=mask,
    max_words=300,
    contour_width=1,
    contour_color="royalblue"
)
wc.generate_from_frequencies(word_tfidf)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
