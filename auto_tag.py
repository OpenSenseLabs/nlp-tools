from flask import Flask, request, render_template, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statistics import mode
import  numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

def extract_NN(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>}

    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne

def sentiment(sent):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(sent)
    max_score = max(score['neu'],max(score['pos'],score['neg']))
    if bool(0.5 > score['compound'] > -0.5 ):
        return "neutral"
    elif score['compound']< -0.5:
        return 'negative'
    else:
        return 'positive'

def extract_ne_type(entities,sent,not_word):
    NE_TYPES = ["ORGANIZATION","PERSON","LOCATION", "DATE","TIME","MONEY","PERCENT","FACILITY","GPE"]
    sentences = nltk.sent_tokenize(sent)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    entity_type=dict()
    #entity_name=set()
    for en in entities:
        if en.lower() not in not_word:
            entity_type[en]="Other"
    for sent in tokenized_sentences:
        parse_tree = nltk.ne_chunk(nltk.tag.pos_tag(sent), binary=False)
        for i in parse_tree.subtrees(): # finding type of entity
            if i.label() in NE_TYPES:
                name= ' '.join([child[0] for child in i.leaves()])
                entity_type[name.strip()] = i.label()
    #print entity_type
    return entity_type


from string import punctuation
def find_occurrence(sent,ne):
    cnt=0
    i=sent.find(ne,0)
    while i!=-1:
        cnt+=1
        i=sent.find(ne,i+len(ne))
    return cnt


def calculate_wight(entities,text):
    weight=dict()
    #print entities
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1, 10))
    tfidf_mat= vectorizer.fit_transform([text])
    stop_word=vectorizer.get_stop_words()
    #entity_type = extract_ne_type(entities,text,stop_word)
    #print entity_type
    for word,w8 in zip(vectorizer.get_feature_names(),tfidf_mat.toarray().tolist()[0]):
        weight[word]=w8
    spcl_c=['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\'',"]","^",'_',"`","{","|","}","~"]
    tpl_list=set()
    for word in entities:
        if word.lower() not in stop_word:
            ss=word.lower()
            for ch in spcl_c:
                ss=ss.replace(ch,' ')
            ss= ' '.join([w.strip() for w in ss.split(' ') if len(w.strip())>1 and w.strip() not in stop_word])
            try:
                tpl_list.add((word.lower(), weight[ss]))#, entity_type[word]))
            except KeyError as e:
                continue
    return tpl_list

def extract(text):
    entities = dict()

    for sent in nltk.sent_tokenize(text):
        senti=sentiment(sent)
        for ne in extract_NN(sent):
            try:
                entities.append(senti)
            except:
                entities[ne]=[senti]
            
    not_word=[]
    for ne in entities:
        if len(ne)<3:
            not_word.append(ne)
            continue
        entities[ne]=mode(entities[ne])
    for i in not_word:
        del entities[i]
    response =dict()
    for (ne,weight) in calculate_wight([str(entity) for entity in entities], text):
        response[ne.lower()] = {'weight':weight}#, 'sentiment':entities[ne]}
    
    return jsonify(response)

@app.route('/keywords', methods = ['POST'])
def my_form_post():
	text = request.form['text']
	processed_text = extract(text)
	return processed_text

if(__name__ == "__main__"):
	app.run(port = 5000)