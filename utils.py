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

def clean_target_level_input(level, model='cefr') -> int:
    if model=='cefr':
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
    elif model=='catile':
        if not isinstance(level,dict) or len(level)!=1 or list(level.keys())[0] not in ['catile','age','grade']:
            raise InformError('target_level should be a dictionary with only one key: "catile", "age", or "grade".')
        if not isinstance(list(level.values())[0],(int,float)):
            raise InformError('The value should be a number.')
        key = list(level.keys())[0]
        value = list(level.values())[0]

        grade2catile = [-160,170,430,650,850,950,1030,1100,1160,1210,1250,1300,1300]
        if key=='age':
            grade = min(max(0,value-5),12)
            return grade2catile[grade]
        elif key=='grade':
            grade = min(max(0,value),12)
            return grade2catile[grade]
        else:
            return int(round(value/10)*10)
            