from flask import Flask, request
from flask_cors import CORS, cross_origin

import spacy
import en_core_web_lg

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
import nltk
import heapq

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'

nltk.download('stopwords')
nltk.download('punkt')

stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_lg')


@app.route('/summarise', methods=['POST'])
@cross_origin()
def summarise():
    payload = request.get_json()
    raw_text = payload['text']

    normalised_text = normalize_text(raw_text)
    cleaned_text = cleanup_text(normalised_text)
    return {"summary": generate_summary(raw_text, cleaned_text[0]),
            "image": "https://gph.is/28TEXx4"
            }


@app.route('/', methods=['GET'])
def index():
    return "Container is running"


def normalize_text(text):
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm1, flags=re.DOTALL)
    return tm3.replace("\n", "")


def cleanup_text(docs, logging=False):
    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return pd.Series(texts)


def generate_summary(text_without_removing_dot, cleaned_text):
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list = []
    for idx, sentence in enumerate(doc.sents):  # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]', '', str(sentence)))

    word_frequencies = {}
    for word in nltk.word_tokenize(cleaned_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print("Original Text:\n")
    print(text_without_removing_dot)
    print('\n\nSummarized text:\n')
    print(summary)
    return summary
