import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import spacy, re, textacy, pickle, pronouncing
from textacy import text_stats
import pandas as pd
from spacy.lang.en import English
from spacy.tokens import Span
import Levenshtein as lev
from lexical_diversity import lex_div as ld
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import numpy as np
from tensorflow import keras
from scipy.optimize import curve_fit, fsolve
from g2p_en import G2p

# Rasch model functions
def rasch_func(x, a, b, c):
    return c + ((1-c) / (1+np.exp(a*(b-x))))

def rasch_func_solve(x, arr):
    a = arr[0]
    b = arr[1]
    c = arr[2]
    y = arr[3]
    return c + ((1-c) / (1+np.exp(a*(b-x)))) - y

def rasch_results(x, arr):
    a = arr[0]
    b = arr[1]
    c = arr[2]
    y = c + ((1-c) / (1+np.exp(a*(b-x))))
    return y

def high_avg(arr):
    uniques = np.unique(arr)
    highest_val = uniques[-1]

    total = 0
    counter = 1
    len_counter = 0
    for x in uniques:
        cnt = np.count_nonzero(arr == x)
        total += x*cnt*counter
        len_counter += cnt*counter
        counter += 1

    return total / len_counter

def double_high_avg(arr):
    uniques = np.unique(arr)
    highest_val = uniques[-1]

    total = 0
    counter = 1
    len_counter = 0
    for x in uniques:
        cnt = np.count_nonzero(arr == x)
        total += x*cnt*counter
        len_counter += cnt*counter
        counter = counter * 2

    return total / len_counter

punct_re = re.compile(r'\s([, \.;\?])')

def standardize_2(text):
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
    text = re.sub('\s+',' ',text)
    text = re.sub(punct_re, r'\g<1>', text)
    
    return text


# ab_at -> ability at X%
# Ex: ab_at = 0.75 -> ability at 75%
# ab_at can be an array, like [0.5, 0.75, 0.9]
def get_ability_2 (data, ab_at_arr, n_bins = 'auto', n_iter = 1):
    # sort the requested abilities
    ab_at_arr = np.sort(ab_at_arr)
    
    # remove all zeros
    data = [i for i in data if i != 0]

    if str(n_bins) != 'auto':
        h, data_hist = np.histogram(data, bins = n_bins, density=True)  
    else:
        h, data_hist = np.histogram(data, bins = 'auto', density=True)
   
    h_csum = np.cumsum(h)
    h_csum_norm = h_csum / h_csum.max()

    est_x = np.mean(data_hist)

    init_vals = [1,est_x,0]
    best_vals, covar = curve_fit(rasch_func, data_hist[0:-1], h_csum_norm, p0=init_vals)

#     print('best_vals', best_vals)
#     print('covar', covar)

    ability = {}
    
    for ab_at in ab_at_arr:
        vals = list(best_vals)
        vals.append(ab_at)
        the_ab = fsolve(rasch_func_solve, [est_x], vals)
        ability[ab_at] = the_ab[0]

    return ability



# ab_at -> ability at X%
# Ex: ab_at = 0.75 -> ability at 75%
# ab_at can be an array, like [0.5, 0.75, 0.9]
def get_ability(arr, ab_at):
    # sort the requested abilities
    ab_at = np.sort(ab_at)
    
    # remove all zeros
    arr = [i for i in arr if i != 0]
    
#     print(arr)
    
#     if len(arr) <= 10:
#         h, the_ab = np.histogram(arr, bins = len(arr), density=True)
#     else:
    h, the_ab = np.histogram(arr, bins = 'auto', density=True)
    h_csum = np.cumsum(h)
    h_csum_norm = h_csum / h_csum.max()
    
    n_ab = len(ab_at)
    
    ability = {}
    counter = 0
    
#     print('hcsum')
#     print(h_csum_norm)
    
    for x in range(len(h_csum_norm)):
        
        if counter < n_ab:
            temp_ab = h_csum_norm[x]
            if temp_ab >= ab_at[counter]:
                ability[ab_at[counter]] = the_ab[x]
                counter += 1
        else:
            break
    
#     while len(ability) < n_ab:
        
#     if len(ability) < 2:
#         ability[0.75] = the_ab[-1]
#     if len(ability) < 3:
#         ability[0.9] = the_ab[-1]
    
    return ability


def sim_from_doc(doc):
    sentences = [s for s in doc.sents]
    sims = []
    for x in range(len(sentences)):
        for y in range(x+1, len(sentences)):
            sims.append(sentences[x].similarity(sentences[y]))
    return (1 - np.mean(sims))

def get_ability_3(arr, ab_at):
    # sort the requested abilities
    ab_at = np.sort(ab_at)
    
    # remove all zeros
    arr = [i for i in arr if i != 0]
    
    n_bins = len(arr) * 2
    
    h, the_ab = np.histogram(arr, bins = n_bins, density=True)
    h_csum = np.cumsum(h)
    h_csum_norm = h_csum / h_csum.max()
    
    ability = {}
    
    end_point = len(h_csum_norm)
    next_round_point = 0
    for x in ab_at:
        start_point = next_round_point
        for y in range(start_point, end_point):
            next_round_point = y
            
            temp_ab = h_csum_norm[y]
            if temp_ab >= x:
                ability[x] = the_ab[y]
                break
        
        if next_round_point == end_point - 1:
            ability[x] = the_ab[next_round_point]
    
    return ability

# DECODABILITY
dipthong_phones = ['AW', 'OY', 'AO']
vowel = ['a', 'e', 'i', 'o', 'u']

decode_degree_dict = {
    'vce': 4,
    'cvce': 4,
    'ccvce': 4,
    'cccvce': 4,
    'cvr': 6,
    'ccvr': 6,
    'vrc': 6,
    'cvrc': 6,
    'ccvrc': 6,
    'vll': 6,
    'cvll': 6,
    'ccvll': 6,
    'cvlc': 6,
    'ccvlc': 6,
    'cvvlc': 6,
    'ccvvlc': 6,
    'v': 1,
    'cv': 1,
    'cvc': 2,
    'vc': 2,
    'cce': 3,
    'ccv': 3,
    'vcc': 3,
    'vccc': 3,
    'ccvc': 3,
    'cccvc': 3,
    'cvcc': 3,
    'cvccc': 3,
    'ccvcc': 3,
    'ccvccc': 3,
    'cccvcc': 3,
    'cccvccc': 3,
    'cvv': 5,
    'cve': 5,
    'ccve': 5,
    'cvvc': 5,
    'cvvcc': 5,
    'ccvv': 5,
    'ccvvc': 5,
    'ccvvcc':5,
    'vvc': 5,
    'vvcc': 5
}

def decode_str(s):
    s = s.lower()
    decoding = ''
    for l in s:
        if l in vowel:
            decoding += 'v'
        else:
            decoding += 'c'
            
    if len(s) >= 3:
        # words that end with 'e'
        if s[-1] == 'e':
            decoding = decoding[0:-1] + 'e'
            
        # words with 'r' as a second last char
        if s[-2] == 'r' and decoding[-3] == 'v':
            decoding = decoding[0:-2] + 'r' + decoding[-1]

        # words with 'l' as a second last char
        elif s[-2] == 'l' and decoding[-3] == 'v' and decoding[-1] == 'c':
            decoding = decoding[0:-2] + 'l' + decoding[-1]
            
    if len(s) >= 2:
        if s[-1] == 'y':
            decoding = decoding[0:-1] + 'v'
            
        # words that end with 'r'
        elif s[-1] == 'r':
            decoding = decoding[0:-1] + 'r'
        
        # words that end with 'll'
        if s[-2:] == 'll':
            decoding = decoding[0:-2] + 'll'

        # words that end with 'gh'
        elif s[-2:] == 'gh':
            decoding = decoding[0:-2] + 'gh'

    return decoding

def any_dipthongs(s):
    is_there = False
    raw_phones = pronouncing.phones_for_word(s)
    
    if len(raw_phones):
        phones = raw_phones[0].split()
        for phone in phones:
            # remove numbers
            phone = ''.join([i for i in phone if not i.isdigit()])

            if phone in dipthong_phones:
                is_there = True
                break
    return is_there


def decoding_level(s):
    decode = decode_str(s)
    degree = decode_degree_dict.get(decode, 9)
    
    if any_dipthongs(s):
        if degree != 6:
            degree = 7
    
    return degree

def count_syllables(s):

    phonomes = G2p(s)
    counter = 0
    for phonome in phonomes:
        if any(char.isdigit() for char in phonome):
            counter += 1

    return counter


def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def lcs(strings):
    substr = ''
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0])-i+1):
                if j > len(substr) and all(strings[0][i:i+j] in x for x in strings):
                    substr = strings[0][i:i+j]
    return len(substr)/max([len(x) for x in strings])

def get_noncompressibility(text):
    def encode(text):
        array = []
        for i in range(len(text)):
            st = text[i:] + text[0:i]  # generating cyclic rotations
            array.append(st)
        array.sort()  # sorting the cyclic rotations according to unicode values
        bwt = ''
        for i in range(len(array)):
            bwt += array[i][-1]
        return bwt

    def compress(string):
        compressed = ""
        count = 1
        #Add in first character
        compressed += string[0]
        #Iterate through loop, skipping last one
        for i in range(len(string)-1):
            if(string[i] == string[i+1]):
                count+=1
            else:
                if(count > 1):
                    #Ignore if no repeats
                    compressed += str(count)
                compressed += string[i+1]
                count = 1
        #print last one
        if(count > 1):
            compressed += str(count)
        return compressed
    return len(compress(encode(text)))/len(text)


def get_tense(x):
    proceed = False
    confident = True
    tense = []
    
    if sum([child.lemma_ == 'to' for child in x.children if child.i<x.i])==0:
        if x.head == x or x.dep_ == 'ROOT':
            proceed = True
        elif x.pos_ != 'AUX':
            if sum([child.pos_ == 'AUX' for child in x.children if child.i<x.i])>0 or x.tag_ != x.head.tag_:
                proceed = True
            else:
                x = x.head
                confident = False
                proceed = True
        if proceed == True:
            for child in x.children:
                if child.dep_ in ['aux','auxpass'] and child.i<x.i:
                    tense.append(child.orth_.lower())
                    if child.lemma_ == 'to':
                        return None
            
            if x.tag_ in ['VBZ','VBP','VB']: #do/does
                tense.append('do')
            elif x.tag_ == 'VBD': #did
                tense.append('did')
            elif len(tense) > 0:
                if x.tag_ == 'VBG': #doing
                    tense.append('doing')
                elif x.tag_ == 'VBN': #done
                    tense.append('done')
                if confident == False:
                    tense = ['*']+tense
    if len(tense)==0:
        return None
    else:
        tense = ' '.join(tense).replace("’","'").replace("wo ",'will ').replace("ca ",'can ').replace(
            "'s done",'has done').replace("'s doing",'is doing').replace("'s doing",'is doing').replace(
                "'d done",'had done').replace("'d do",'would do').replace("'m",'am').replace("'ve",'have').replace('do do','do').replace('does do','does').replace('did do','did')
        return tense


def process(text):

    text = standardize_2(text)
    doc = nlp(text)

    #run_time('nlp')
    
    # Split text
    n_words_temp = 0
    last_start_index = np.nan
    for sent in reversed(list(doc.sents)):
        n_words_temp += text_stats.basics.n_words(sent)
        if n_words_temp>=125:
            last_start_index = sent[0].i
            break
    #run_time('Split text')
    
    # Levenshtein distance
    lev_distance_list = []
    sent_list = [sent.text.lower().strip(' ').strip('\n') for sent in list(doc.sents) if len(sent.text.strip(' ').strip('\n'))>1]
    if len(sent_list) >= 2:
        for i in range(1, len(sent_list)):
            lev_distance_list.append(lev.distance(sent_list[i-1],sent_list[i]))
    else:
        lev_distance_list.append(0)
    #run_time('Levenshtein distance')
    
    dfs = []
    dfs_500 = []
    summaries = []
    sent_depth_list = []
    sent_length_list2 = []
    ending_index = np.nan
    n_sents = 0
    root_id = None
    
    sent_length_list = []
    sent_list = []
    lcs2_list = []
    lcs3_list = []
    aoa_list = []
    original_list = []
    stop_list = []
    pos_list = []
    morph_list = []
    tag_list = []
    tense_list = []
    lemma_list = []
    dep_list = []
    ttr_list = []
    sent_index_list = []

    freq_list = []
    freq_list_no_stop = []
    abstract_list = []
    nsyl_list = []
    
    decode_lemma_list = []
    decode_original_list = []
    
    n_words = 0
    
    for x in doc:
        # Lemmatize pronouns
        if x.lemma_ == '-PRON-':
            if x.orth_ == 'I':
                x.lemma_ = x.orth_
            else:
                x.lemma_ = x.orth_.lower()
        # Lemmatize would
        if x.lemma_ in ["'d","’d"] and x.pos_ == 'PART':
            x.lemma_ = 'would'
            x.pos_ = 'VERB'
            x.tag_ = 'MD'
            x.dep_ = 'aux'
        # Lemmatize special plurals
        if x.tag_ in ['NNPS','NNS']:
            if x.orth_.lower() in no_singular.iloc[:,0].values:
                x.lemma_ = x.orth_.lower()

        if x.orth_.lower() in mismatches.word.values and x.pos_ in mismatches.pos.values:
            x.lemma_ = mismatches_dict[x.orth_.lower()]
        
        original_list.append(x.orth_)
        
        dep_list.append(x.dep_)
        pos_list.append(x.pos_)
        morph_list.append(spacy.explain(x.tag_))
        
        if x.lemma_ == 'an':
            x.lemma_ = 'a'
        elif x.lemma_ in ["n't","n’t", "not"]:
            x.lemma_ = 'not'
            pos_list[-1] = 'ADV'
        elif x.lemma_ in ["'ll","’ll"]:
            x.lemma_ = 'will'
            if pos_list[-1] == 'VERB':
                pos_list[-1] = 'AUX'
        elif x.lemma_ in ["'d","’d"] and morph_list[-1] == 'verb, modal auxiliary':
            x.lemma_ = 'would'
            if pos_list[-1] == 'VERB':
                pos_list[-1] = 'AUX'
        elif x.lemma_ in ["'","’"] and morph_list[-1] == 'verb, 3rd person singular present':
            x.lemma_ = 'be'
        elif x.lemma_ in ["'ve","’ve"]:
            x.lemma_ = 'have'
        elif x.lemma_ in ["'s","’s"] and pos_list[-1] == 'PRON':
            x.lemma_ = 'us'
        elif x.lemma_ in ["'","’"] and pos_list[-1] == 'PART':
            x.lemma_ = 'us'
            pos_list[-1] = 'PRON'
        elif x.lemma_ in ["ca","Ca"] and pos_list[-1] == 'AUX':
            x.lemma_ = 'can'
            if pos_list[-1] == 'VERB':
                pos_list[-1] = 'AUX'
        elif x.lemma_ in ["sha","Sha"] and pos_list[-1] == 'AUX':
            x.lemma_ = 'shall'
            if pos_list[-1] == 'VERB':
                pos_list[-1] = 'AUX'
        elif x.lemma_ in ["ai","Ai"] and pos_list[-1] == 'AUX':
            if spacy.explain(x.head.tag_) == 'verb, past participle':
                x.lemma_ = 'have'
            else:
                x.lemma_ = 'be'
        elif x.lemma_ in ["to"]:
            pos_list[-1] = 'ADP'
        elif x.lemma_ in ["can", "could", "will", "would", "should", "may", "might", "must", "shall", "ought", "cannot"] and pos_list[-1] == 'VERB':
            pos_list[-1] = 'AUX'
        elif x.lemma_ in ["Can", "Could", "Will", "Would", "Should", "May", "Might", "Must", "Shall", "Ought", "Cannot"] and pos_list[-1] == 'VERB':
            pos_list[-1] = 'AUX'

        lemma_list.append(x.lemma_.lower())
        tag_list.append(x.tag_)
        if pos_list[-1] in ['VERB','AUX']:
            tense_list.append(get_tense(x))
        else:
            tense_list.append(None)


        # Summarize

        if lemma_list[-1] in set(stopwords.words('english')):
            stop_list.append(True)
        else:
            stop_list.append(False)

        # Frequency
        try:
            index = corpus_words.index(lemma_list[-1].lower())
            freq_list.append(corpus_freq[index])
            if x.is_stop:
                freq_list_no_stop.append(0)
            else:
                freq_list_no_stop.append(corpus_freq[index])
        except:
            freq_list.append(0)
            freq_list_no_stop.append(0)

        # AOA
        try:
            index = aoa_words.index(lemma_list[-1].lower())
            aoa_list.append(aoa[index])
        except:
            aoa_list.append(0)

        # ABSTRACTNESS
        try:
            index = abstract_words.index(lemma_list[-1].lower())
            abstract_list.append(abstract[index])
        except:
            abstract_list.append(0)

        # NUMBER OF SYLLABLES and DECODABILITY
        the_nsyl = count_syllables(original_list[-1].lower())
#         print('lemma', lemma_list[-1].lower())
#         print('nsyl', the_nsyl)
        nsyl_list.append(the_nsyl)
        if the_nsyl > 2:
            decode_lemma_list.append(9)
            decode_original_list.append(9)
        else:
            decode_lemma_list.append(decoding_level(lemma_list[-1].lower()))
            decode_original_list.append(decoding_level(original_list[-1].lower()))
#         try:
# #             the_nsyl = textacy.TextStats(x).n_syllables
#             # The code above is throwing error
#             # Using MRC database
# #             index = nsyl_words.index(lemma_list[-1].lower())
# #             the_nsyl = nsyl[index]
# #             nsyl_list.append(the_nsyl)
#             if the_nsyl > 2:
#                 dec_list.append(9)
#             else:
#                 dec_list.append(decoding_level(lemma_list[-1].lower()))
#         except:
# #             nsyl_list.append(0)
#             dec_list.append(decoding_level(lemma_list[-1].lower()))
            
            
        if x.sent.root.i != root_id:
            root_id = x.sent.root.i
            n_sents += 1
        sent_index_list.append(n_sents-1)
        
        if x.dep_ == 'ROOT' and x.pos_ not in ['SPACE','PUNCT']:
            if len(x.sent.text.strip(' ').strip('\n'))>1:
                sent_list.append(x.sent.text.lower().strip(' ').strip('\n'))
            sent_depth_list.append(walk_tree(x.sent.root, 0))
            n_words += text_stats.basics.n_words(x.sent)
#             sent_length_list.append(textacy.TextStats(x.sent).n_syllables)
            sent_length_list.append(count_syllables(str(x.sent)))
            ending_index = x.sent[-1].i

        if n_words >= 125 or pd.isnull(last_start_index) or x.i == len(doc)-1:
            if ending_index < last_start_index or x.i == len(doc)-1:
                
                df = pd.DataFrame({'original':original_list,
                                   'lemma':lemma_list,
                                   'frequency':freq_list,
                                   'aoa':aoa_list,
                                   'abstract':abstract_list,
                                   'nsyl': nsyl_list,
                                   'pos':pos_list,
                                   'dep':dep_list,
                                   'tag':tag_list,
                                   'tense': tense_list,
                                   'decode_lemma': decode_lemma_list,
                                   'decode_original': decode_original_list,
                                   'is_stop':stop_list,
                                   'sent_index':sent_index_list})
                #run_time('frequency')
                
                noncompressibility = get_noncompressibility(' '.join(df['original'].values).replace('\n',' ').replace('  ',' ').strip(' '))
                #run_time('noncompressibility')
                
                df = df[(df['pos']!='SPACE')&(df['pos']!='PUNCT')]

                tokens = df[df['pos']!='NUM']['lemma'].values # Need to retrain the model with NUM later
                hdd = ld.hdd(tokens)
                mtld = ld.mtld_ma_wrap(tokens)
                maas_ttr = ld.maas_ttr(tokens)

                df_freq_type = df[df['frequency']!=0].drop_duplicates(['lemma','pos'])
#                 mean_log_freq_type = np.log(df_freq_type['frequency']).sum()/len(df_freq_type)
                mean_log_freq_type = df_freq_type['frequency'].sum()/len(df_freq_type)
                
                df_freq_type_no_stop = df[(df['frequency']!=0)&(~df['is_stop'])].drop_duplicates(['lemma','pos'])
#                 mean_log_freq_type_no_stop = np.log(df_freq_type_no_stop['frequency']).sum()/len(df_freq_type_no_stop)
                mean_log_freq_type_no_stop = df_freq_type_no_stop['frequency'].sum()/len(df_freq_type_no_stop)
                
                df_freq_token = df[df['frequency']!=0]
#                 mean_log_freq_token = np.log(df_freq_token['frequency']).sum()/len(df_freq_token)
                mean_log_freq_token = df_freq_token['frequency'].sum()/len(df_freq_token)
                
                df_freq_token_no_stop = df[(df['frequency']!=0)&(~df['is_stop'])]
#                 mean_log_freq_token_no_stop = np.log(df_freq_token_no_stop['frequency']).sum()/len(df_freq_token_no_stop)
                mean_log_freq_token_no_stop = df_freq_token_no_stop['frequency'].sum()/len(df_freq_token_no_stop)
                #run_time('hdd & mtld & maas_ttr')
                
                # Longest common string
                for i in range(len(sent_list)):
                    couplet = sent_list[i:min(i+2,len(sent_list))]
                    triplet = sent_list[i:min(i+3,len(sent_list))]
                    if len(couplet) == 2:
                        lcs2_list.append(lcs(couplet))
                    if len(triplet) == 3:
                        lcs3_list.append(lcs(triplet))

                #run_time('Longest common string')
                if len(lcs2_list) == 0:
                    lcs2_list = [0]
                if len(lcs3_list) == 0:
                    lcs3_list = [0]
                
                summary = pd.Series({'mean_length':np.mean(sent_length_list),
                                     'mean_log_freq_type':mean_log_freq_type,
                                     'mean_log_freq_type_no_stop':mean_log_freq_type_no_stop,
                                     'mean_log_freq_token':mean_log_freq_token,
                                     'mean_log_freq_token_no_stop':mean_log_freq_token_no_stop,
                                     'maas_ttr':maas_ttr,
                                     'hdd':hdd,
                                     'mtld':mtld,
                                     'mean_lev_distance':np.mean(lev_distance_list),
                                     'mean_lcs2':np.mean(lcs2_list),
                                     'mean_lcs3':np.mean(lcs3_list),
                                     'noncompressibility':noncompressibility})
                summary.name = 'counts'

                dfs.append(df)
                summaries.append(summary)
                
                if len(dfs)%4 == 0:
                    dfs_500.append(pd.concat(dfs))
                
                sent_length_list2 += sent_length_list
    
                sent_length_list = []
                sent_list = []
                lcs2_list = []
                lcs3_list = []
                aoa_list = []
                original_list = []
                stop_list = []
                pos_list = []
                tag_list = []
                morph_list = []
                lemma_list = []
                dep_list = []
                tense_list = []
                freq_list = []
                ttr_list = []
                sent_index_list = []
                freq_list = []
                freq_list_no_stop = []
                abstract_list = []
                nsyl_list = []
                decode_lemma_list = []
                decode_original_list = []
                n_words = 0
    
    
    df_data = pd.concat(dfs)
    if len(dfs)%4 != 0:
        dfs_500.append(pd.concat(dfs[-(len(dfs)%4):]))
        
    density_list = []
    for df_500 in dfs_500:
        df_500 = df_500[~df_500['is_stop']]
        vectorizer = TfidfVectorizer(smooth_idf=True)
        X = vectorizer.fit_transform([' '.join(x[1]['lemma'].values.astype(str)) for x in df_500.groupby('sent_index')])
        if len(df_500['lemma'].unique()) > 1:
            try:
                svd_model = TruncatedSVD(n_components=min(X.shape[1]-1,min(len(df_500['lemma'].unique())-1,10)), algorithm='arpack', n_iter=100, random_state=0)
                svd_model.fit(X)
            except:
                svd_model = TruncatedSVD(n_components=min(X.shape[1]-1,min(len(df_500['lemma'].unique())-1,10)), algorithm='randomized', n_iter=100, random_state=0)
                svd_model.fit(X)
            density_list.append(svd_model.explained_variance_ratio_.max())
        else:
            density_list.append(np.nan)
        
#     print(sent_length_list2)
    summary = pd.concat([pd.Series({'std_length':np.std(sent_length_list2),
                                    'high_mean_length': high_avg(sent_length_list2),
                                    'mean_depth':np.mean(sent_depth_list),
                                    'std_depth':np.std(sent_depth_list),
                                    'density':np.mean(density_list)}),
                         sum(summaries)/len(summaries)]).sort_index()
    
    
    summary['n_sents'] = len(list(doc.sents))
    
    # Mean syllable count
    df_data2 = df_data.copy()
    df_data2['original'] = df_data2['original'].apply(lambda x: x.lower())
    
    nsyl_clean_list = df_data2[df_data2['nsyl'] != 0].drop_duplicates('original')['nsyl'].tolist()
    summary['nsyl_mean'] = np.mean(nsyl_clean_list)
    
    # High mean of the syllable count
    summary['nsyl_high_mean'] = high_avg(nsyl_clean_list)
    
    
    # Decoding Demand
    summary['decode_lemma'] = np.mean(df_data[df_data['decode_lemma'] != 0]['decode_lemma'].tolist())
    summary['decode_original'] = np.mean(df_data[df_data['decode_original'] != 0]['decode_original'].tolist())
    
    
    abstract_score_list = df_data[df_data['abstract'] != 0].drop_duplicates('lemma')['abstract'].tolist()
    
    if len(abstract_score_list) == 0:
        summary['abstract_mean'] = 0
        summary['abstract_high_mean'] = 0
    else:
        summary['abstract_mean'] = np.mean(abstract_score_list)
        summary['abstract_high_mean'] = high_avg(abstract_score_list)
#     n_bins = len(abst_score_list) * 2
#     abst_ability = get_ability_2(abst_score_list, ability_at, n_bins=n_bins)
#     summary['abst_50'] = abst_ability[0.5]
#     summary['abst_75'] = abst_ability[0.75]
#     summary['abst_90'] = abst_ability[0.9]
    
    
    # RASCH MODEL ABILIES AT X%
#     ability_at = [0.5, 0.75, 0.9]
    ability_at = np.arange(0.2,0.96,0.01)
    
    aoa_clean_list = df_data[df_data['aoa'] != 0].drop_duplicates('lemma')['aoa'].tolist()
    
    if len(aoa_clean_list) == 0:
        summary['aoa_mean'] = 0
        summary['aoa_high_mean'] = 0
        
        for a in ability_at:
            summary['aoa_' + str(int(a*100))] = 0
    else:
        summary['aoa_mean'] = np.mean(aoa_clean_list)
        summary['aoa_high_mean'] = high_avg(aoa_clean_list)
        
        try:
            aoa_ability = get_ability_2(aoa_clean_list, ability_at)
            for a in ability_at:
                summary['aoa_' + str(int(a*100))] = aoa_ability[a]
        except:
            aoa_ability = get_ability_3(aoa_clean_list, ability_at)
            for a in ability_at:
                summary['aoa_' + str(int(a*100))] = aoa_ability[a]
    
    
    freq_stoplist_df = df_data[df_data['frequency'] != 0].drop_duplicates('lemma')
    freq_stoplist_df = freq_stoplist_df[~freq_stoplist_df.lemma.isin(most_freq_50)]
    freq_clean_list = freq_stoplist_df['frequency'].tolist()
    
    # 12.6 is the highest value on the frequency list
    # 13 is used as the closest int to 12.6
    rareness_clean_list = (13-freq_stoplist_df['frequency']).tolist()
    
    if len(freq_clean_list) == 0:
        # set the values to the lowest rareness
        summary['freq_mean'] = 0.5
        summary['freq_high_mean'] = 0.5
        
        for a in ability_at:
            summary['freq_' + str(int(a*100))] = 0.5
    else:
        summary['freq_mean'] = np.mean(rareness_clean_list)
        summary['freq_high_mean'] = high_avg(rareness_clean_list)

        try:
            freq_ability = get_ability_2(rareness_clean_list, ability_at)
            for a in ability_at:
                summary['freq_' + str(int(a*100))] = freq_ability[a]
        except:
            freq_ability = get_ability_3(rareness_clean_list, ability_at)
            for a in ability_at:
                summary['freq_' + str(int(a*100))] = freq_ability[a]
    
    return df_data, summary

def Lexile2level(Lexile, a=300, b=-100, c=840):
    # a: level to be Lexile 450
    # b: Lexile level to be 0
    # c: Lexile level to be 666
    if Lexile<450:
        level=(Lexile-b)/(450-b)*a
    else:
        level=(Lexile-450)/(c-450)*(666-a)+a
    return level

def kde_predict(summary, kde_model):
    x = summary[['density','hdd','mean_depth','mean_lcs2','mean_lcs3','mean_length','high_mean_length','mean_lev_distance','mean_log_freq_token_no_stop','mean_log_freq_type',
          'mtld','noncompressibility','std_depth','std_length','nsyl_high_mean','decode_original','abstract_high_mean']+
         ['aoa_{}'.format(x) for x in range(80,95,1)]+['freq_{}'.format(x) for x in range(80,95,1)]].values
    y_pred = []
    candidates = []
    for y in np.arange(-140,1260,10):
        candidates.append([y,kde_model.score([np.append(x,y)])])
    #y_pred.append([pd.DataFrame(candidates).sort_values(1,ascending=False)[0].iloc[:5].mean()])
    return pd.DataFrame(candidates).sort_values(1,ascending=False)[0].iloc[:5].mean()

def neural_predict(summary, neural_model, transformer):
    summary = summary[['density','hdd','mean_depth','mean_lcs2','mean_lcs3','mean_length','high_mean_length','mean_lev_distance','mean_log_freq_token_no_stop','mean_log_freq_type',
              'mtld','noncompressibility','std_depth','std_length','nsyl_high_mean','decode_original','abstract_high_mean']+
             ['aoa_{}'.format(x) for x in range(45,96,1)]+['freq_{}'.format(x) for x in range(45,96,1)]].values
    X = transformer.transform([summary])
    return neural_model.predict(X)[0][0]

def is_outlier(summary, outlier_ranges):
    _filter1 = []
    _filter2 = []
    for k in outlier_ranges.keys():
        if k in ['high_mean_length','mean_lev_distance','noncompressibility', 'std_depth','std_length','abstract_high_mean']:
            _filter1.append(summary[k]<outlier_ranges[k][0] or summary[k]>outlier_ranges[k][1])
        else:
            _filter2.append(summary[k]<outlier_ranges[k][0] or summary[k]>outlier_ranges[k][1])
    return np.sum(_filter1),np.sum(_filter2)

def custom_loss(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = loss*(y_true/450+1)
    loss = K.sum(loss, axis=1)
    return loss

def predict(summary, neural_model, transformer, kde_model, outlier_ranges):
    # n_outliers1: number of outliers among 6 variables
    # n_outliers2: number of outliers among aoa and freq 45-95
    # neural_model weight = (4-n_outliers1)/4, kde model weight = n_outliers1/4
    # if n_outliers2 !=0, use only kde_model
    n_outliers1, n_outliers2 = is_outlier(summary, outlier_ranges)
    y_pred_neural = neural_predict(summary,neural_model,transformer)
    y_pred_kde = kde_predict(summary,kde_model)
    pred_lexile = (y_pred_neural*((max(0,4-n_outliers1))/4)+y_pred_kde*(min(n_outliers1,4)/4)) * (n_outliers2==0) + y_pred_kde * (n_outliers2!=0)
    return pred_lexile

def predict_all(df, neural_model, transformer, kde_model, outlier_ranges, lexile=True):
    results = []
    for i in range(len(df)):
        summary = df.iloc[i]
        n_outliers1, n_outliers2 = is_outlier(summary, outlier_ranges)
        y_pred_neural = neural_predict(summary,neural_model,transformer)
        y_pred_kde = kde_predict(summary,kde_model)
        pred_lexile = (y_pred_neural*((max(0,4-n_outliers1))/4)+y_pred_kde*(min(n_outliers1,4)/4)) * (n_outliers2==0) + y_pred_kde * (n_outliers2!=0)
        if lexile:
            results.append([y_pred_neural,y_pred_kde,pred_lexile,n_outliers1, n_outliers2])
        else:
            results.append([Lexile2level(y_pred_neural),Lexile2level(y_pred_kde),Lexile2level(pred_lexile),n_outliers1, n_outliers2])
        print('\r','{}%'.format(round((i+1)/len(df)*100,1)),end='')
    df_pred = pd.DataFrame(results)
    df_pred.columns = ['neural','kde','combined','n_outliers1', 'n_outliers2']
    #return pd.concat([df.iloc[:,:3].reset_index(drop=True),df_pred],axis=1)
    return df_pred

def initialize():
    global g2p,nlp,no_singular,mismatches,mismatches_dict,corpus_words,corpus_freq,most_freq_50, aoa_words, aoa, abstract_words, abstract, nsyl_words, nsyl, neural_model, kde_model, transformer, outlier_ranges

    g2p = G2p()
    #spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("sentencizer", before="parser")
    
    # NO SINGULAR
    no_singular = pd.read_csv('files/model_files/noun_pluralia_tantum.csv')


    # SPACY MISMATCH
    mismatches = pd.read_csv('files/model_files/mismatches.csv')
    mismatches_dict = {}
    for x in range(len(mismatches)):
        mismatches_dict[mismatches.iloc[x]['word']] = mismatches.iloc[x]['lemma']

        
    # FREQUENCY
    df_corpus = pd.read_excel('files/model_files/Corpus_frequency_log.xlsx')
    corpus_words = list(df_corpus['word'].values)
    corpus_freq = list(df_corpus['TOTAL'].values)

    # 50 most frequenct words
    # most_freq_50 = df_corpus.sort_values('TOTAL', ascending=False)['word'][0:50].to_list()
    most_freq_50 = set(stopwords.words('english'))


    # aoa
    df_aoa = pd.read_excel('files/model_files/AoA_ratings_Kuperman_et_al_BRM.xlsx')

    # remove nan values
    df_aoa = df_aoa[~df_aoa['Rating.Mean'].isna()]
    # shortlist the 50 most frequent words
    df_aoa = df_aoa[~df_aoa.Word.isin(most_freq_50)]
    aoa_words = list(df_aoa['Word'].values)
    aoa = list(df_aoa['Rating.Mean'].values)


    # MRC PSYCHOLINGUISTIC DATABASE

    # The file includes the following data
    #   -concreteness
    #   -imagability
    #   -number of syllables
    #   -number of phonomes
    df_mrc = pd.read_csv('files/model_files/mrc_database_cnc_img_nsyl_nphon.csv')

    # imagability
    # later changed into abstractness
    # abstractness = 700 - img
    df_abstract = df_mrc[df_mrc['img'] != 0]

    # shotlist the 50 most frequent words
    df_abstract = df_abstract[~df_abstract.word.isin(most_freq_50)]
    abstract_words = list(df_abstract['word'].values)

    # abstractness = 700 - img
    abstract = list(700 - df_abstract['img'].values)


    # number of syllables
    df_nsyl = df_mrc[df_mrc['nsyl'] != 0]

    # shotlist the 50 most frequent words
    df_nsyl = df_nsyl[~df_nsyl.word.isin(most_freq_50)]
    nsyl_words = list(df_nsyl['word'].values)
    nsyl = list(df_nsyl['nsyl'].values)


    neural_model = keras.models.load_model(r'files/model_files/model94_2.h5',custom_objects={'custom_loss': custom_loss})
    kde_model = pickle.load(open(r'files/model_files/kde_model.pkl', 'rb'))
    transformer = pickle.load(open(r'files/model_files/normalizer.pkl', 'rb'))
    outlier_ranges = pickle.load(open(r'files/model_files/outlier_ranges.pkl', 'rb'))


def get_difficulty(text, lexile=False):
    df, summary = process(text)
    return predict_all(pd.DataFrame(summary).T.fillna(0), neural_model, transformer, kde_model, outlier_ranges, lexile=lexile)