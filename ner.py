import time
from collections import Counter
from datetime import datetime, timedelta
from urllib.request import urlopen
import numpy as np
from bs4 import BeautifulSoup  

import spacy
# !python -m spacy download en_core_web_sm
import en_core_web_sm
spacy_nlp = spacy.load('en_core_web_sm')
from spacy import displacy
spacy.prefer_gpu()
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

import warnings
warnings.filterwarnings('ignore')

parragraphs = {}    
article_text = {}
counter = 1
count_p = 1
    
page = urlopen('https://finance.yahoo.com/news').read()
soup = BeautifulSoup(page, features="html.parser")
posts = soup.findAll("div", {"class": "Cf"})

for post in posts:
    
    time.sleep(2)
    url = post.a['href']
    print("Article:", counter, "URL:", url)
    whole_url = 'https://finance.yahoo.com'+url
    try:
        link_page = urlopen(whole_url).read()         
    except:
        link_page = urlopen(url).read()

    link_soup = BeautifulSoup(link_page)
    passa = link_soup.findAll("p")
    passage = ""

    for sentence_p in passa:
        passage += sentence_p.text
        parragraphs[str(counter)+ '-'+str(count_p)] = sentence_p.text
        count_p += 1
    article_text[counter] = passage
    counter +=1

articles = {}

for k,v in article_text.items():
    articles[k] = v 

#print(parragraphs.keys())
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def extract_currency_relations(doc):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    # Named Entity Extraction and Relation Extraction /// Building Key/Value pairs for numeric values
    relations = {}
    for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
        if not money.is_stop:
            if money.dep_ in ("attr", "dobj"):
                subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    subject = subject[0]
                    relations[subject] = money
            elif money.dep_ == "pobj" and money.head.dep_ == "prep":
                relations[money.head.head] = money

    for quantity in filter(lambda w: w.ent_type_ == "QUANTITY", doc):
        if not quantity.is_stop:
            if quantity.dep_ in ("attr", "dobj"):
                subject = [w for w in quantity.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    subject = subject[0]
                    relations[subject] = quantity
            elif quantity.dep_ == "pobj" and quantity.head.dep_ == "prep":
                relations[quantity.head.head] = quantity 

    return relations

words = []

#Events Identification - Using pattern i added new specific type of entity
nlp = English()
ruler = EntityRuler(nlp)
pattern = [{"label": "ORG", "pattern": "Apple"}]
ruler.add_patterns(pattern)
nlp.add_pipe(ruler)

first_art =[]
second_art = []
third_art = []
fourth_art = []
fifth_art = []
sixth_art = []

first_art_p =[]
second_art_p = []
third_art_p = []
fourth_art_p = []
fifth_art_p = []
sixth_art_p = []

#Named Entity Extraction
for text in parragraphs:
    doc = spacy_nlp(parragraphs[text])
    relations = extract_currency_relations(doc)
    if '1-' in text: 
        first_art_p.append(parragraphs[text])
        for r1 in relations:
            #print("{:<10}\t{}\t{}".format(r1.text, r1.ent_type_,relations[r1].text))
            words.append(r1.lemma_)
            first_art.append(r1.lemma_)
    if '2-' in text:  
        second_art_p.append(parragraphs[text]) 
        for r1 in relations:
            words.append(r1.lemma_)
            second_art.append(r1.lemma_)
    if '3-' in text:  
        third_art_p.append(parragraphs[text]) 
        for r1 in relations:
            words.append(r1.lemma_)
            third_art.append(r1.lemma_)
    if '4-' in text:   
        fourth_art_p.append(parragraphs[text])
        for r1 in relations:
            words.append(r1.lemma_)
            fourth_art.append(r1.lemma_)
    if '5-' in text:   
        fifth_art_p.append(parragraphs[text])
        for r1 in relations:
            words.append(r1.lemma_)
            fifth_art.append(r1.lemma_)
    if '6-' in text:   
        sixth_art_p.append(parragraphs[text])
        for r1 in relations:
            words.append(r1.lemma_)
            sixth_art.append(r1.lemma_)

# Count frequency of appearance in currently analyzed articles
word_freq = Counter(words)
common_words = word_freq.most_common(5)
#print (common_words, len(words))

#Sentiment Analysis
# Analyze each paragraph of the article and provide sentiment score for it based on Financial context of it.
# TF-IDF (Term Frequency-Inverse Document Frequency)
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')

txt_fitted = tf.fit(words)

txt_transformed = txt_fitted.transform(words)
#print(tf.vocabulary_)
print(np.mean(tf.fit_transform(first_art).toarray()))
print(np.mean(tf.fit_transform(second_art).toarray()))
print(np.mean(tf.fit_transform(third_art).toarray()))
print(np.mean(tf.fit_transform(fourth_art).toarray()))
print(np.mean(tf.fit_transform(fifth_art).toarray()))
print(np.mean(tf.fit_transform(sixth_art).toarray()))

print(tf.fit_transform(first_art_p).toarray())
print(tf.fit_transform(second_art_p).toarray())
print(tf.fit_transform(third_art_p).toarray())
print(tf.fit_transform(fourth_art_p).toarray())
print(tf.fit_transform(fifth_art_p).toarray())
print(tf.fit_transform(sixth_art_p).toarray())







