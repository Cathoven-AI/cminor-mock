import re
import numpy as np

# Replace Chinese punctuation marks with English ones
# Get rig of double space
# Leave a space around quotation marks for better sentence splitting
def standardize(text):
    text = text.replace('\ufeff','')
    text = text.replace("\n", " ")
    text = re.sub("’","'",text)
    text = re.sub("‘","'",text)
    text = re.sub("“",'"',text)
    text = re.sub("”",'"',text)
    text = re.sub("！","!",text)
    text = re.sub("？","?",text)
    text = re.sub("，",",",text)
    text = re.sub("；",";",text)
    text = re.sub("（","(",text)
    text = re.sub("）",")",text)
    text = re.sub("【","[",text)
    text = re.sub("】","]",text)
    text = re.sub('"',' " ',text)
    text = re.sub("[ ]{2,}", " ",text)
    return text


def sim_spacy_cross_sent(doc):
    sentences = [s for s in doc.sents]
    sims = []
    for x in range(len(sentences)):
        for y in range(x+1, len(sentences)):
            sims.append(sentences[x].similarity(sentences[y]))
    return np.mean(sims)