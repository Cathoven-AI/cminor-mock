import pandas as pd
import re

import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

mismatches = pd.read_csv(os.path.join(BASE_DIR, 'files/model_files/mismatches.csv'))
mismatches_dict = mismatches.set_index(['word','pos'])['lemma'].to_dict()


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



def fine_lemmatize(x,doc,spacy):
    if x.lemma_.startswith('-'):
        x.lemma_ = x.lemma_.strip('-')

    # Lemmatize special plurals
    #if x.tag_ in set(['NNPS','NNS']):
    #    if x.orth_.lower() in no_singular.iloc[:,0].values:
    #        x.lemma_ = x.orth_.lower()

    if x.orth_ == 'the':
        x.pos_ == 'DET'
    elif x.lemma_ == 'an':
        x.lemma_ = 'a'
    elif x.lemma_ in set(['got','gotten']):
        x.lemma_ = 'get'
    elif x.lemma_ in set(["n't","n’t", "not"]):
        x.lemma_ = 'not'
        x.pos_ = 'ADV'
    elif x.lemma_ in set(["'ll","’ll"]):
        x.lemma_ = 'will'
        if x.pos_ == 'VERB':
            x.pos_ = 'AUX'
    elif x.orth_.lower() in set(["'d","’d"]) and x.i+1<len(doc):
        if x.nbor(1).tag_!="VB":
            x.lemma_ = 'have'
        elif x.tag_ == 'MD':
            if x.nbor(1).orth_.lower() == 'better':
                x.lemma_ = 'have'
            else:
                x.lemma_ = 'would'
            if x.pos_ == 'VERB':
                x.pos_ = 'AUX'
        elif x.pos_ == 'PART':
            x.lemma_ = 'would'
            x.pos_ = 'VERB'
            x.tag_ = 'MD'
            x.dep_ = 'aux'
    elif x.orth_.lower() == "'t":
        x.lemma_ = 'it'
        x.pos_ = 'PRON'
    elif x.lemma_ in set(["'","’"]):
        if x.pos_ == 'PART':
            if x.i>=0 and doc[x.i-1].pos_ not in set(['NOUN','PROPN']):
                x.lemma_ = 'us'
                x.pos_ = 'PRON'
            else:
                x.lemma_ = "'s"
        elif spacy.explain(x.tag_) == 'verb, 3rd person singular present':
            x.lemma_ = 'be'
    elif x.lemma_ in set(["'ve","’ve"]):
        x.lemma_ = 'have'
    elif x.lemma_ in set(["'s","’s"]) and x.pos_ == 'PRON':
        x.lemma_ = 'us'
    elif x.lemma_ in set(["ca","Ca"]) and x.pos_ == 'AUX':
        x.lemma_ = 'can'
    elif x.lemma_ in set(["sha","Sha"]) and x.pos_ == 'AUX':
        x.lemma_ = 'shall'
    elif x.lemma_ in set(["ai","Ai"]) and x.pos_ == 'AUX':
        if spacy.explain(x.head.tag_) == 'verb, past participle':
            x.lemma_ = 'have'
        else:
            x.lemma_ = 'be'
    #elif x.lemma_ in set(["to"]):
    #    x.pos_ = 'ADP'
    elif x.lemma_ in set(["can", "could", "will", "would", "should", "may", "might", "must", "shall", "ought", "cannot",
                      "Can", "Could", "Will", "Would", "Should", "May", "Might", "Must", "Shall", "Ought", "Cannot"]) and x.pos_ == 'VERB':
        x.pos_ = 'AUX'
    elif x.orth_.lower() in set(['me','him','her','us',"them"]):
        x.lemma_ = x.orth_.lower()
    elif x.lemma_ == 'wilde' and x.pos_=='VERB':
        x.lemma_ = 'wild'

    if x.lemma_ == 'be' and x.i==x.head.i:
        x.pos_ = 'VERB'
        
    x.lemma_ = mismatches_dict.get((x.orth_.lower(),x.pos_),x.lemma_.lower())
    
    if x.pos_.endswith('CONJ'):
        x.pos_ = 'CONJ'
    elif x.pos_=='VERB' and '-' in x.orth_:
        x.lemma_ = '-'.join(x.orth_.split('-')[:-1]+[lemmatizer.lemmatize(x.orth_.split('-')[-1],'v')])
    elif x.pos_ == "PUNCT" and len(re.findall(r'[A-Za-z]', x.orth_))>=6:
        x.pos_ = "ADJ"

    if x.lemma_=='' and x.orth_!='':
        x.lemma_ = x.orth_

    no_hyphen = x.lemma_.strip('-')
    if no_hyphen!='':
        x.lemma_ = no_hyphen
    #if x.lemma_.endswith('.') and not x.orth_[0].isupper() and not '.' in x.lemma_.strip('.'):
    #    x.lemma_ = x.lemma_.strip('.')
    return x