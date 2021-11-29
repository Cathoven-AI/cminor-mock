import numpy as np

# Calculates the longes common string
def lcs(strings):
    substr = ''
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0])-i+1):
                if j > len(substr) and all(strings[0][i:i+j] in x for x in strings):
                    substr = strings[0][i:i+j]
    return len(substr)/max([len(x) for x in strings])


# Couplet LCS
def lcs_2(doc):
    sent_list = [sent.text.lower().strip(' ').strip('\n') for sent in doc.sents]
    lcs2_list = []
    for i in range(len(sent_list)):
        couplet = sent_list[i:min(i+2,len(sent_list))]
        if len(couplet) == 2:
            lcs2_list.append(lcs(couplet))

    return len


# Triplet LCS
def lcs_3(doc):
    sent_list = [sent.text.lower().strip(' ').strip('\n') for sent in doc.sents]
    lcs3_list = []
    for i in range(len(sent_list)):
        triplet = sent_list[i:min(i+3,len(sent_list))]
        if len(triplet) == 3:
            lcs3_list.append(lcs(triplet))


# Couplet and triplet LCS together
def lcs_2_3(doc):
    sent_list = [sent.text.lower().strip(' ').strip('\n') for sent in doc.sents]
    lcs2_list = []
    lcs3_list = []
    for i in range(len(sent_list)):
        couplet = sent_list[i:min(i+2,len(sent_list))]
        triplet = sent_list[i:min(i+3,len(sent_list))]
        if len(couplet) == 2:
            lcs2_list.append(lcs(couplet))
        if len(triplet) == 3:
            lcs3_list.append(lcs(triplet))

    return np.mean(lcs2_list), np.mean(lcs3_list)


# Longest commons string comparing all sentences with each other
def lcs_cross_sent(doc):
    sent_list = [sent.text.lower().strip(' ').strip('\n') for sent in doc.sents]
    cross_sent_lcs_list = []
    for i in range(len(sent_list)):
        for j in range(i+1, len(sent_list)):
            cross_sent_lcs_list.append(lcs([sent_list[i], sent_list[j]]))
            
    return np.mean(cross_sent_lcs_list)


# Check the percentage
# (num of words from the first sent that is in the second) / (num of words of the first sent)
def lcs_words_cross_sent(doc):
    sent_tokens = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            if not (token.is_punct or token.pos_ =='SPACE'):
                tokens.append(token.lemma_.lower())
        if len(tokens) > 1:
            sent_tokens.append(tokens)

    structure_sim = []
    #print(sent_tokens)
    for x in range(len(sent_tokens)):
        tokens1 = sent_tokens[x]
        for y in range(x+1, len(sent_tokens)):
            tokens2 = sent_tokens[y]
            structure_sim.append(sum([t in tokens2 for t in tokens1]) / len(tokens1))

    #print(structure_sim)
            
    return np.mean(structure_sim)
    