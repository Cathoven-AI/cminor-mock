import pandas as pd

import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

mismatches = pd.read_csv(os.path.join(BASE_DIR, 'files/model_files/mismatches.csv'))
mismatches_dict = {}
for x in range(len(mismatches)):
    mismatches_dict[mismatches.iloc[x]['word']] = mismatches.iloc[x]['lemma']


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



def fine_lemmatize(x,doc,spacy):
    if x.lemma_.startswith('-'):
        x.lemma_ = x.lemma_.strip('-')

    # Lemmatize special plurals
    #if x.tag_ in ['NNPS','NNS']:
    #    if x.orth_.lower() in no_singular.iloc[:,0].values:
    #        x.lemma_ = x.orth_.lower()

    if x.orth_.lower() in mismatches.word.values and x.pos_ in mismatches.pos.values:
        x.lemma_ = mismatches_dict[x.orth_.lower()]

    if x.lemma_ == 'an':
        x.lemma_ = 'a'
    elif x.lemma_ in ['got','gotten']:
        x.lemma_ = 'get'
    elif x.lemma_ in ["n't","n’t", "not"]:
        x.lemma_ = 'not'
        x.pos_ = 'ADV'
    elif x.lemma_ in ["'ll","’ll"]:
        x.lemma_ = 'will'
        if x.pos_ == 'VERB':
            x.pos_ = 'AUX'
    elif x.orth_.lower() in ["'d","’d"]:
        if 'Aspect=Perf' in doc[min(x.i+1,len(doc)-1)].morph:
            x.lemma_ = 'have'
        elif x.tag_ == 'MD':
            if doc[min(x.i+1,len(doc)-1)].orth_.lower() == 'better':
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
            
    elif x.lemma_ in ["'","’"] and spacy.explain(x.tag_) == 'verb, 3rd person singular present':
        x.lemma_ = 'be'
    elif x.lemma_ in ["'ve","’ve"]:
        x.lemma_ = 'have'
    elif x.lemma_ in ["'s","’s"] and x.pos_ == 'PRON':
        x.lemma_ = 'us'
    elif x.lemma_ in ["'","’"] and x.pos_ == 'PART':
        x.lemma_ = 'us'
        x.pos_ = 'PRON'
    elif x.lemma_ in ["ca","Ca"] and x.pos_ == 'AUX':
        x.lemma_ = 'can'
    elif x.lemma_ in ["sha","Sha"] and x.pos_ == 'AUX':
        x.lemma_ = 'shall'
    elif x.lemma_ in ["ai","Ai"] and x.pos_ == 'AUX':
        if spacy.explain(x.head.tag_) == 'verb, past participle':
            x.lemma_ = 'have'
        else:
            x.lemma_ = 'be'

    #elif x.lemma_ in ["to"]:
    #    x.pos_ = 'ADP'
    elif x.lemma_ in ["can", "could", "will", "would", "should", "may", "might", "must", "shall", "ought", "cannot",
                      "Can", "Could", "Will", "Would", "Should", "May", "Might", "Must", "Shall", "Ought", "Cannot"] and x.pos_ == 'VERB':
        x.pos_ = 'AUX'

    if x.lemma_ == 'be' and x.i==x.head.i:
        x.pos_ = 'VERB'
        
    x.lemma_ = x.lemma_.lower()
    
    if x.pos_.endswith('CONJ'):
        x.pos_ = 'CONJ'
    elif x.pos_=='VERB' and '-' in x.orth_:
        x.lemma_ = '-'.join(x.orth_.split('-')[:-1]+[lemmatizer.lemmatize(x.orth_.split('-')[-1],'v')])

    #if x.lemma_.endswith('.') and not x.orth_[0].isupper() and not '.' in x.lemma_.strip('.'):
    #    x.lemma_ = x.lemma_.strip('.')
    return x