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
    text = text.replace("\n", " ")
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

class InformError(Exception):
    def __init__(self, message):
        super().__init__(str(message))

def check_level_input_and_int(level) -> int:
    level_str2int = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
    if isinstance(level,str):
        if level.upper() not in level_str2int:
            raise InformError("level should be one of 'A1', 'A2', 'B1', 'B2', 'C1', 'C2'.")
        level = level_str2int[level.upper()]
    elif isinstance(level,(int,float)):
        if level<0 or level>=6:
            raise InformError("level should be between 0 and 5.")
        level = int(level)
    else:
        raise InformError("level should be one of 'A1', 'A2', 'B1', 'B2', 'C1', 'C2' or a number between 0 and 5.")
    return level