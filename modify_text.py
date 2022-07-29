import re
import numpy as np

def standardize(text):
    text = standardize_patch(text)
    text = text.replace('[',' [')
    text = text.replace('(',' (')
    text = text.replace('’',"'")
    text = text.replace('‘',"'")
    text = text.replace('´',"'")
    text = text.replace('“','"')
    text = text.replace('˝','"')
    text = text.replace('”','"')
    text = text.replace('!','! ')
    text = text.replace('?','? ')
    text = re.sub(r'\.(?=[^ \W\d])', '. ', text)

    return text


def standardize_patch(text):
    text = text.replace('\ufeff','')
    text = re.sub("！","!",text)
    text = re.sub("？","?",text)
    text = re.sub("，",",",text)
    text = re.sub("；",";",text)
    text = re.sub("（","(",text)
    text = re.sub("）",")",text)
    text = re.sub("【","[",text)
    text = re.sub("】","]",text)
    return text


# Replace Chinese punctuation marks with English ones
# Get rig of double space
# Leave a space around quotation marks for better sentence splitting
def standardize_old(text):
    text = text.replace('\ufeff','')
    text = text.replace("\n", " \n")
    text = text.replace("\r", " ")
    text = re.sub("’","'",text)
    text = re.sub("‘","'",text)
    text = text.replace('´',"'")
    text = re.sub("“",'"',text)
    text = re.sub("”",'"',text)
    text = text.replace('˝','"')
    text = re.sub("！","!",text)
    text = re.sub("？","?",text)
    text = re.sub("，",",",text)
    text = re.sub("；",";",text)
    text = re.sub("（","(",text)
    text = re.sub("）",")",text)
    text = re.sub("【","[",text)
    text = re.sub("】","]",text)
    text = text.replace('[',' [')
    text = text.replace('(',' (')
    text = re.sub('"',' " ',text)
    #text = re.sub("[ ]{2,}", " ",text)
    return text


def remove_extra_spaces(text):
    text = re.sub(' " ','"',text)
    text = text.replace(' [', '[')
    text = text.replace(' (', '(')

    return text


def sim_spacy_cross_sent(doc):
    sentences = [s for s in doc.sents]
    sims = []
    for x in range(len(sentences)):
        for y in range(x+1, len(sentences)):
            sims.append(sentences[x].similarity(sentences[y]))
    return np.mean(sims)