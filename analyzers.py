import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from . import spacy
from . import word as solar_word
from . import modify_text
from .edit_distance_modified import edit_distance
import pickle, re, tensorflow, textstat, warnings, openai
from textacy import text_stats
from collections import Counter
import Levenshtein as lev
from lexical_diversity import lex_div as ld
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import curve_fit, fsolve
from scipy.stats import percentileofscore
from .lemmatizers import fine_lemmatize
from nltk.stem import WordNetLemmatizer, LancasterStemmer, PorterStemmer
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Lexile files
neural_model = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/lexile_20220410_2.h5'),compile=False)

with open(os.path.join(BASE_DIR, 'files/model_files/lexile_linear.pkl'), 'rb') as file:
    linear_model = pickle.load(file)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/nsyl_high_mean_arr.npy'), 'rb') as f:
    nsyl_high_mean_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/decode_original_arr.npy'), 'rb') as f:
    decode_original_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/freq_high_mean_arr.npy'), 'rb') as f:
    freq_high_mean_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_log_freq_token_no_stop_arr.npy'), 'rb') as f:
    mean_log_freq_token_no_stop_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_log_freq_type_arr.npy'), 'rb') as f:
    mean_log_freq_type_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/aoa_high_mean_arr.npy'), 'rb') as f:
    aoa_high_mean_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/abstract_high_mean_arr.npy'), 'rb') as f:
    abstract_high_mean_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_length_arr.npy'), 'rb') as f:
    mean_length_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_lev_distance_arr.npy'), 'rb') as f:
    mean_lev_distance_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_lcs2_arr.npy'), 'rb') as f:
    mean_lcs2_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mean_lcs3_arr.npy'), 'rb') as f:
    mean_lcs3_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/mtld_arr.npy'), 'rb') as f:
    mtld_arr = np.load(f)

with open(os.path.join(BASE_DIR, 'files/model_files/normalization_arrays_2022.05.25/hdd_arr.npy'), 'rb') as f:
    hdd_arr = np.load(f)

br2am = pd.read_excel(os.path.join(BASE_DIR, 'files/model_files/br2am_2021.04.25.xlsx'))
br2am_dict = dict(zip(br2am.british, br2am.american))

df_corpus = pd.read_excel(os.path.join(BASE_DIR, 'files/model_files/Corpus_frequency_log.xlsx'))
corpus_words = df_corpus['word'].to_list()
corpus_freq = df_corpus['TOTAL'].to_list()
most_freq_50 = set(stopwords.words('english'))

df_aoa = pd.read_excel(os.path.join(BASE_DIR, 'files/model_files/AoA_ratings_Kuperman_et_al_BRM.xlsx'))
# remove nan values
df_aoa = df_aoa[~df_aoa['Rating.Mean'].isna()]
# shortlist the 50 most frequent words
df_aoa = df_aoa[~df_aoa.Word.isin(most_freq_50)]
aoa_words = list(df_aoa['Word'].values)
aoa = df_aoa['Rating.Mean'].to_list()

df_mrc = pd.read_csv(os.path.join(BASE_DIR, 'files/model_files/mrc_database_cnc_img_nsyl_nphon.csv'))
df_abstract = df_mrc[df_mrc['img'] != 0]
# shotlist the 50 most frequent words
df_abstract = df_abstract[~df_abstract.word.isin(most_freq_50)]
abstract_words = df_abstract['word'].to_list()
abstract = list(700 - df_abstract['img'].values)

# CEFR files
cefr_w_pos_min_prim = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_w_pos_min_prim.pkl'),'rb'))
cefr_wo_pos_min_prim = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_wo_pos_min_prim.pkl'),'rb'))
cefr_w_pos_mean_prim = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_w_pos_mean_prim.pkl'),'rb'))
cefr_wo_pos_mean_prim = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_wo_pos_mean_prim.pkl'),'rb'))

cefr_w_pos = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_w_pos.pkl'),'rb'))
cefr_wo_pos = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_wo_pos.pkl'),'rb'))
cefr_w_pos_sup = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_w_pos_sup.pkl'),'rb'))
cefr_wo_pos_sup = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/cefr_wo_pos_sup.pkl'),'rb'))

df_reference_words = pd.concat([pd.DataFrame(pd.Series(cefr_w_pos_min_prim)).reset_index(),pd.DataFrame(pd.Series(cefr_w_pos)).reset_index()]).drop_duplicates(['level_0','level_1'])
df_temp = df_reference_words[df_reference_words['level_0'].apply(lambda x: str(x).lower().endswith('e') and len(str(x))>=4)].copy()
df_temp['level_0'] = [x[:-1] for x in df_temp['level_0'].values]
df_reference_words = pd.concat([df_reference_words,df_temp]).drop_duplicates(['level_0','level_1']).reset_index(drop=True)

df_reference_words_sup = pd.DataFrame(pd.Series(cefr_w_pos_sup)).reset_index()
df_temp = df_reference_words_sup[df_reference_words_sup['level_0'].apply(lambda x: str(x).lower().endswith('e') and len(str(x))>=4)].copy()
df_temp['level_0'] = [x[:-1] for x in df_temp['level_0'].values]
df_reference_words_sup = pd.concat([df_reference_words_sup,df_temp]).drop_duplicates(['level_0','level_1']).reset_index(drop=True)

cefr_word_model = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/cefr_word_model.h5'))


df_phrases = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/phrases.pkl'),'rb'))
df_phrases = df_phrases[~((df_phrases['characters']<=9)&(df_phrases['length']<=2)&(df_phrases['level']<=1)&df_phrases['word'].apply(lambda x: x in set(['have','and','do','it','or','on','so','at','you','after','in','down','i','up','that','to'])))][['id','original','clean','followed_by','lemma','pos','word','is_idiom','ambiguous','phrase_parts']]
phrase_original2id = df_phrases.set_index('original')['id'].to_dict()
people_list = set(pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/people_list.pkl'),'rb')))


del br2am, df_corpus, df_aoa, df_mrc, df_abstract, df_temp


nlp = spacy.load('en_core_web_trf')

stemmers = [PorterStemmer(), LancasterStemmer()]
lemmatizer = WordNetLemmatizer()

gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "files/model_files/gpt_tokenizer"))


class AdoTextAnalyzer(object):
    def __init__(self, openai_api_key=None):
        self.text = None
        self.doc = None
        self.cefr = None
        self.readability = None
        self.catile = None
        self.simplifier = None
        self.adaptor = None
        self.openai_api_key = openai_api_key

    def analyze_cefr(self,text,propn_as_lowest=True,intj_as_lowest=True,keep_min=True,
                    return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                    return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                    return_clause_stats=True,return_phrase_count=True,return_final_levels=True,return_result=False,clear_simplifier=True):
        text = self.clean_text(text)
        if text!=self.text:
            self.doc = None
            self.cefr = None
            self.readability = None
            self.catile = None
            if clear_simplifier:
                self.simplifier = None
                self.adaptor = None
            self.text = text

        temp_settings = {'propn_as_lowest':propn_as_lowest,'intj_as_lowest':intj_as_lowest,'keep_min':keep_min,
                        'return_sentences':return_sentences, 'return_wordlists':return_wordlists,'return_vocabulary_stats':return_vocabulary_stats,
                        'return_tense_count':return_tense_count,'return_tense_term_count':return_tense_term_count,'return_tense_stats':return_tense_stats,'return_clause_count':return_clause_count,
                        'return_clause_stats':return_clause_stats,'return_phrase_count':return_phrase_count,'return_final_levels':return_final_levels}

        if self.cefr is None or temp_settings!=self.cefr.print_settings():
            if self.doc is None:
                self.doc = nlp(self.text)
                for x in self.doc:
                    x = fine_lemmatize(x,self.doc,nlp)
            self.cefr = self.CefrAnalyzer(self)
            self.cefr.start_analyze(propn_as_lowest,intj_as_lowest,keep_min,
                        return_sentences, return_wordlists,return_vocabulary_stats,
                        return_tense_count,return_tense_term_count,return_tense_stats,return_clause_count,
                        return_clause_stats,return_phrase_count,return_final_levels)
        if return_result:
            return self.cefr.result

    def analyze_readability(self,text,language='en',return_result=False):
        text = self.clean_text(text)
        if text!=self.text:
            self.doc = None
            self.cefr = None
            self.readability = None
            self.catile = None
            self.simplifier = None
            self.adaptor = None
            self.text = text
        if self.readability is None:
            self.readability = self.ReadabilityAnalyzer(self)
            self.readability.start_analyze(language)
        if return_result:
            return self.readability.result

    def analyze_catile(self,text,return_result=False):
        text = self.clean_text(text)
        if text!=self.text:
            self.doc = None
            self.cefr = None
            self.readability = None
            self.catile = None
            self.simplifier = None
            self.adaptor = None
            self.text = text
        if self.catile is None:
            if self.doc is None:
                self.doc = nlp(self.text)
                for x in self.doc:
                    x = fine_lemmatize(x,self.doc,nlp)
            self.catile = self.CatileAnalyzer(self)
            self.catile.start_analyze()
        if return_result:
            return self.catile.result

    def simplify(self, text, target_level, target_adjustment=0.5, n=1, by_sentence=False, auto_retry=False, up=False, return_result=False):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key
        text = self.clean_text(text)
        self.doc = None
        self.cefr = None
        self.readability = None
        self.catile = None
        self.simplifier = None
        self.adaptor = None
        self.text = text

        self.simplifier = self.CefrSimplifier(self)
        self.simplifier.start_simplify(text, target_level, target_adjustment=target_adjustment, n=n, by_sentence=by_sentence, auto_retry=auto_retry, up=up)

        if return_result:
            return self.simplifier.result

    def adapt(self, text, target_level, target_adjustment=0.5, even=False, n=1, auto_retry=False, return_result=False):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key
        text = self.clean_text(text)
        self.doc = None
        self.cefr = None
        self.readability = None
        self.catile = None
        self.simplifier = None
        self.adaptor = None
        self.text = text

        self.adaptor = self.CefrAdaptor(self)
        self.adaptor.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, auto_retry=auto_retry)

        if return_result:
            return self.adaptor.result

    def clean_text(self, text):
        return text.replace("\u00A0", " ").replace('\xa0',' ').strip()

    class CatileAnalyzer(object):

        def __init__(self, outer):
            self.shared_object = outer
            self.result = None

        def walk_tree(self, node, depth):
            if node.n_lefts + node.n_rights > 0:
                return max(self.walk_tree(child, depth + 1) for child in node.children)
            else:
                return depth

        def get_noncompressibility(self, text):
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
                if len(string):
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
            
            if len(text):
                return len(compress(encode(text)))/len(text)
            else:
                return 0

        def lcs(self,strings):
            substr = ''
            if len(strings) > 1 and len(strings[0]) > 0:
                for i in range(len(strings[0])):
                    for j in range(len(strings[0])-i+1):
                        if j > len(substr) and all(strings[0][i:i+j] in x for x in strings):
                            substr = strings[0][i:i+j]
            return len(substr)/max([len(x) for x in strings])

        def high_avg(self, arr):
            uniques = np.unique(arr)

            if len(arr) == 0:
                return 0
            
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

        def get_ability_2(self, data, ab_at_arr, n_bins = 'auto', n_iter = 1):
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

            est_x = np.nanmean(data_hist)

            init_vals = [1,est_x,0]
            best_vals, covar = curve_fit(self.rasch_func, data_hist[0:-1], h_csum_norm, p0=init_vals)

            ability = {}
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'The iteration is not making good progress')
                for ab_at in ab_at_arr:
                    vals = list(best_vals)
                    vals.append(ab_at)
                    the_ab = fsolve(self.rasch_func_solve, [est_x], vals)
                    ability[ab_at] = the_ab[0]

            return ability

        def get_ability_3(self, arr, ab_at):
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

        def rasch_func(self, x, a, b, c):
            return c + ((1-c) / (1+np.exp(a*(b-x))))

        def rasch_func_solve(self, x, arr):
            a = arr[0]
            b = arr[1]
            c = arr[2]
            y = arr[3]
            return c + ((1-c) / (1+np.exp(a*(b-x)))) - y

        def process_text(self):

            #text = standardize_2(text)
            doc = self.shared_object.doc
            #run_time('nlp')
            
            # Split text
            n_words_temp = 0
            last_start_index = np.nan
            for sent in reversed(list(doc.sents)):
                n_words_temp += text_stats.api.TextStats(sent).n_words
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
            
            longest_length = 0
            longest_sent = ""
            
            min_freq = -1.67
            
            for x in doc:
                if bool(re.search('[a-zA-Z]', x.sent.text)):

                    #x = fine_lemmatize(x,doc,nlp)
                    
                    original_list.append(x.orth_)
                    
                    dep_list.append(x.dep_)
                    pos_list.append(x.pos_)
                    morph_list.append(nlp.explain(x.tag_))

                    lemma_list.append(x.lemma_)
                    
                    # Summarize

                    if lemma_list[-1] in set(stopwords.words('english')) or x.is_stop:
                        stop_list.append(True)
                    else:
                        stop_list.append(False)

                    # Frequency
                    try:
                        if lemma_list[-1] in br2am_dict and br2am_dict[lemma_list[-1]] in corpus_words:
                            index = corpus_words.index(br2am_dict[lemma_list[-1]])
                        else:
                            index = corpus_words.index(lemma_list[-1])
                        freq_list.append(corpus_freq[index])
                        if x.is_stop:
                            freq_list_no_stop.append(min_freq)
                        else:
                            freq_list_no_stop.append(corpus_freq[index])
                    except Exception as e:
                        #print(str(e))
                        if not bool(re.match("[a-z]",lemma_list[-1])):
                            freq_list.append(12.7)
                            freq_list_no_stop.append(12.7)
                        else:
                            freq_list.append(min_freq)
                            freq_list_no_stop.append(min_freq)

                    # AOA
                    try:
                        if lemma_list[-1] in br2am_dict and br2am_dict[lemma_list[-1]] in aoa_words:
                            index = aoa_words.index(br2am_dict[lemma_list[-1]])
                        else:
                            index = aoa_words.index(lemma_list[-1])
                        aoa_list.append(aoa[index])
                    except:
                        if not bool(re.match("[a-z]",lemma_list[-1])):
                            aoa_list.append(0)
                        else:
                            if lemma_list[-1].lower() in corpus_words:
                                aoa_list.append(11)
                            else:
                                aoa_list.append(25)

                    # ABSTRACTNESS
                    try:
                        if lemma_list[-1] in br2am_dict and br2am_dict[lemma_list[-1]] in abstract_words:
                            index = abstract_words.index(br2am_dict[lemma_list[-1]])
                        else:
                            index = abstract_words.index(lemma_list[-1])
                        abstract_list.append(abstract[index])
                    except:
                        if not bool(re.match("[a-z]",lemma_list[-1])):
                            abstract_list.append(0)
                        else:
                            abstract_list.append(250)

                    # NUMBER OF SYLLABLES and DECODABILITY
                    the_nsyl = solar_word.count_syllables(original_list[-1].lower())
            #         print('lemma', lemma_list[-1].lower())
            #         print('nsyl', the_nsyl)
                    nsyl_list.append(the_nsyl)
                    if the_nsyl > 2:
                        decode_lemma_list.append(9)
                        decode_original_list.append(9)
                    else:
                        decode_lemma_list.append(solar_word.decoding_degree(lemma_list[-1].lower()))
                        decode_original_list.append(solar_word.decoding_degree(original_list[-1].lower()))
            #         try:
            # #             the_nsyl = text_stats.api.TextStats(x).n_syllables
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
                        
                        
                    if x.sent.root.i != root_id and bool(re.search('[A-Za-z]',str(x.sent))):
                        root_id = x.sent.root.i
                        n_sents += 1
                    sent_index_list.append(n_sents-1)
                    
                    if x.dep_ == 'ROOT' and x.pos_ not in set(['SPACE','PUNCT','X']):
                        if len(x.sent.text.strip(' ').strip('\n'))>1:
                            sent_list.append(x.sent.text.lower().strip(' ').strip('\n'))
                        sent_depth_list.append(self.walk_tree(x.sent.root, 0))
                        
                        length = text_stats.api.TextStats(x.sent).n_words
                        if length>longest_length:
                            longest_sent = modify_text.remove_extra_spaces(str(x.sent)).strip()
                            longest_length = length
                        
                        n_words += length
            #             sent_length_list.append(text_stats.api.TextStats(x.sent).n_syllables)
                        sent_length_list.append(solar_word.count_syllables(str(x.sent)))
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
                                        'decode_lemma': decode_lemma_list,
                                        'decode_original': decode_original_list,
                                        'is_stop':stop_list,
                                        'sent_index':sent_index_list})
                        #run_time('frequency')
                        
                        noncompressibility = self.get_noncompressibility(' '.join(df['original'].values).replace('\n',' ').replace('  ',' ').strip(' '))
                        #run_time('noncompressibility')
                        
                        df = df[(df['pos']!='SPACE')&(df['pos']!='PUNCT')&(df['pos']!='X')]

                        tokens = df[df['pos']!='NUM']['lemma'].values # Need to retrain the model with NUM later
                        hdd = ld.hdd(tokens)
                        mtld = ld.mtld_ma_wrap(tokens)

                        if len(tokens):
                            maas_ttr = ld.maas_ttr(tokens)
                        else:
                            maas_ttr = 0


                        df_freq_type = df[df['frequency']!=min_freq].drop_duplicates(['lemma','pos'])
        #                 mean_log_freq_type = np.log(df_freq_type['frequency']).sum()/len(df_freq_type)
                        mean_log_freq_type = df_freq_type['frequency'].sum()/len(df_freq_type)
                        
                        df_freq_type_no_stop = df[(df['frequency']!=min_freq)&(~df['is_stop'])].drop_duplicates(['lemma','pos'])
        #                 mean_log_freq_type_no_stop = np.log(df_freq_type_no_stop['frequency']).sum()/len(df_freq_type_no_stop)
                        mean_log_freq_type_no_stop = df_freq_type_no_stop['frequency'].sum()/len(df_freq_type_no_stop)
                        
                        df_freq_token = df[df['frequency']!=min_freq]
        #                 mean_log_freq_token = np.log(df_freq_token['frequency']).sum()/len(df_freq_token)
                        mean_log_freq_token = df_freq_token['frequency'].sum()/len(df_freq_token)
                        
                        df_freq_token_no_stop = df[(df['frequency']!=min_freq)&(~df['is_stop'])]
        #                 mean_log_freq_token_no_stop = np.log(df_freq_token_no_stop['frequency']).sum()/len(df_freq_token_no_stop)
                        mean_log_freq_token_no_stop = df_freq_token_no_stop['frequency'].sum()/len(df_freq_token_no_stop)
                        #run_time('hdd & mtld & maas_ttr')
                        
                        # Longest common string
                        for i in range(len(sent_list)):
                            couplet = sent_list[i:min(i+2,len(sent_list))]
                            triplet = sent_list[i:min(i+3,len(sent_list))]
                            if len(couplet) == 2:
                                lcs2_list.append(self.lcs(couplet))
                            if len(triplet) == 3:
                                lcs3_list.append(self.lcs(triplet))

                        #run_time('Longest common string')
                        if len(lcs2_list) == 0:
                            lcs2_list = [0]
                        if len(lcs3_list) == 0:
                            lcs3_list = [0]
                        
                        summary = pd.Series({'mean_length':np.nanmean(sent_length_list),
                                            'mean_log_freq_type':mean_log_freq_type,
                                            'mean_log_freq_type_no_stop':mean_log_freq_type_no_stop,
                                            'mean_log_freq_token':mean_log_freq_token,
                                            'mean_log_freq_token_no_stop':mean_log_freq_token_no_stop,
                                            'maas_ttr':maas_ttr,
                                            'hdd':hdd,
                                            'mtld':mtld,
                                            'mean_lev_distance':np.nanmean(lev_distance_list),
                                            'mean_lcs2':np.nanmean(lcs2_list),
                                            'mean_lcs3':np.nanmean(lcs3_list),
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
                        morph_list = []
                        lemma_list = []
                        dep_list = []
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

            # If len is less than 1, dfs_500 is an empty frame
            if len(dfs_500) > 1:
                for df_500 in dfs_500:
                    df_500 = df_500[~df_500['is_stop']]
                    vectorizer = TfidfVectorizer(smooth_idf=True)
                    X = vectorizer.fit_transform([' '.join(x[1]['lemma'].values.astype(str)) for x in df_500.groupby('sent_index')])
                    
                    if len(df_500['lemma'].unique()) > 1:
                        svd_model = TruncatedSVD(n_components=min(X.shape[1]-1,min(len(df_500['lemma'].unique())-1,10)), algorithm='randomized', n_iter=100, random_state=0)
                        svd_model.fit(X)
                        density_list.append(svd_model.explained_variance_ratio_.max())
                    else:
                        density_list.append(np.nan)
                
        #     print(sent_length_list2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                summary = pd.concat([pd.Series({'std_length':np.std(sent_length_list2),
                                                'high_mean_length': self.high_avg(sent_length_list2),
                                                'mean_depth':np.nanmean(sent_depth_list),
                                                'std_depth':np.std(sent_depth_list),
                                                'density':np.nanmean(density_list)}),
                                    sum(summaries)/len(summaries)]).sort_index()
            
            
            #summary['n_sents'] = len(list(doc.sents))
            
            # Mean syllable count
            df_data2 = df_data.copy()
            df_data2['original'] = df_data2['original'].apply(lambda x: x.lower())
            
            nsyl_clean_list = df_data2[df_data2['nsyl'] != 0].drop_duplicates('original')['nsyl'].tolist()
            summary['nsyl_mean'] = np.nanmean(nsyl_clean_list)
            
            # High mean of the syllable count
            summary['nsyl_high_mean'] = self.high_avg(nsyl_clean_list)
            
            
            # Decoding Demand
            summary['decode_lemma'] = np.nanmean(df_data[df_data['decode_lemma'] != 0]['decode_lemma'].tolist())
            summary['decode_original'] = np.nanmean(df_data[df_data['decode_original'] != 0]['decode_original'].tolist())
            
            
            abstract_score_list = df_data[df_data['abstract'] != 0].drop_duplicates('lemma')['abstract'].tolist()
            
            if len(abstract_score_list) == 0:
                summary['abstract_mean'] = 0
                summary['abstract_high_mean'] = 0
            else:
                summary['abstract_mean'] = np.nanmean(abstract_score_list)
                summary['abstract_high_mean'] = self.high_avg(abstract_score_list)
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
                summary['aoa_mean'] = np.nanmean(aoa_clean_list)
                summary['aoa_high_mean'] = self.high_avg(aoa_clean_list)
                
                try:
                    aoa_ability = self.get_ability_2(aoa_clean_list, ability_at)
                    for a in ability_at:
                        summary['aoa_' + str(int(a*100))] = aoa_ability[a]
                except:
                    aoa_ability = self.get_ability_3(aoa_clean_list, ability_at)
                    for a in ability_at:
                        summary['aoa_' + str(int(a*100))] = aoa_ability[a]
            
            
            freq_stoplist_df = df_data[df_data['frequency'] != min_freq].drop_duplicates('lemma')
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
                summary['freq_mean'] = np.nanmean(rareness_clean_list)
                summary['freq_high_mean'] = self.high_avg(rareness_clean_list)

                try:
                    freq_ability = self.get_ability_2(rareness_clean_list, ability_at)
                    for a in ability_at:
                        summary['freq_' + str(int(a*100))] = freq_ability[a]
                except:
                    freq_ability = self.get_ability_3(rareness_clean_list, ability_at)
                    for a in ability_at:
                        summary['freq_' + str(int(a*100))] = freq_ability[a]


            summary['n_sents'] = len(df_data.sent_index.unique())
            summary = summary.fillna(0)
            
            return df_data, summary, longest_sent

        def predict(self, summary):
            x = summary[['density','hdd','high_mean_length','maas_ttr','mean_depth','mean_lcs2','mean_lcs3','mean_length','mean_lev_distance','mean_log_freq_token','mean_log_freq_token_no_stop','mean_log_freq_type',
                        'mean_log_freq_type_no_stop','mtld','noncompressibility','std_depth','std_length','nsyl_mean','nsyl_high_mean','decode_original','decode_lemma','abstract_mean','abstract_high_mean','aoa_mean','aoa_high_mean']].values

            y_pred = neural_model.predict(x.reshape(1,-1))[0][0]
            if y_pred>1200:
                y_pred = linear_model.predict([x])[0]
            return y_pred

        def start_analyze(self):
            df, summary, longest_sentence = self.process_text()

            count_lemmas = df.pivot_table(index=['lemma', 'pos'], aggfunc='size')
            
            word_pos = count_lemmas.index.tolist()
            word_count = count_lemmas.values.tolist()
            word_count = sum(word_count)

            #pred_lexile = predict(summary, neural_model, transformer, kde_model, outlier_ranges)
            pred_lexile = self.predict(summary)
            
            #pred_level = Lexile2level(pred_lexile)
            #adjusted_level = adjust_level(pred_level, word_count)

            decoding = 0
            decoding += percentileofscore(nsyl_high_mean_arr, summary['nsyl_high_mean'])
            decoding += percentileofscore(decode_original_arr, summary['decode_original'])
            decoding = decoding / 2

            # mean_log_freq_token_no_stop_arr has inverse proportion relation with Lexile
            # mean_log_freq_type_arr has inverse proportion relation with Lexile
            vocabulary = 0
            vocabulary += percentileofscore(freq_high_mean_arr, summary['freq_high_mean'])
            vocabulary += 100 - percentileofscore(mean_log_freq_token_no_stop_arr, summary['mean_log_freq_token_no_stop'])
            vocabulary += 100 - percentileofscore(mean_log_freq_type_arr, summary['mean_log_freq_type'])
            vocabulary += 3 * percentileofscore(aoa_high_mean_arr, summary['aoa_high_mean'])
            vocabulary += 3 * percentileofscore(abstract_high_mean_arr, summary['abstract_high_mean'])
            vocabulary = vocabulary / 9

            # LCS has inverse proportion relation with Lexile 
            sentences = 0
            sentences += 2 * percentileofscore(mean_length_arr, summary['mean_length'])
            sentences += 2 * percentileofscore(mean_lev_distance_arr, summary['mean_lev_distance'])
            sentences += 100 - percentileofscore(mean_lcs2_arr, summary['mean_lcs2'])
            sentences += 100 - percentileofscore(mean_lcs3_arr, summary['mean_lcs3'])
            sentences = sentences / 6

            patterns = 0
            patterns += percentileofscore(mtld_arr, summary['mtld'])
            patterns += percentileofscore(hdd_arr, summary['hdd'])
            patterns  = patterns / 2

            #normalized_lexile = normalize_lexile(pred_lexile)


            # scores = {"pred_lexile": pred_lexile,
            #         "normalized_lexile": normalized_lexile,
            #         "decodability": decoding,   
            #         "vocab_score": vocabulary,  
            #         "mean_sent_length": sentences,  
            #         "mtld": patterns,
            #         "decoding": decoding,
            #         "vocabulary": vocabulary,
            #         "sentences": sentences,
            #         "patterns": patterns,
            #         "word_count": word_count}

            mean_sent_length = round(word_count/summary['n_sents'],2)

            scores = {
                "catile": pred_lexile,
                "decoding": decoding,
                "vocabulary": vocabulary,
                "sentences": sentences,
                "patterns": patterns
            }



            #return scores

            df_data = df.merge(df.pivot_table(index=['lemma'], aggfunc='size').to_frame(name='lemma_frequency').reset_index(),on='lemma',how='left')
            df_data = df_data[~df_data['is_stop']&(df_data['pos']!='X')&(df_data['pos']!='PROPN')&(df_data['pos']!='AUX')&(df_data['pos']!='PART')].drop_duplicates('lemma')
            df_data = df_data[df_data['lemma'].apply(lambda x:len(x)>=2) & ~((df_data['pos']=='NUM')&(df_data['frequency']<=0))]
            #most_frequent = df_data.sort_values(['lemma_frequency','frequency','aoa','nsyl','decode_lemma'],ascending=[False,True,False,False,False]).head()['lemma'].values
            difficult_words = df_data.sort_values(['frequency','aoa','lemma_frequency','nsyl','decode_lemma'],ascending=[True,False,False,False,False]).head(10)['lemma'].values
            
            self.result = {
                "scores": scores,
                "difficult_words": difficult_words, 
                "longest_sentence": longest_sentence,
                "mean_sent_length": mean_sent_length,
                "word_count": word_count
            }

    class CefrAnalyzer(object):

        def __init__(self,outer):
            self.shared_object = outer

            self.__settings = {'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,
                    'return_sentences':True, 'return_wordlists':True,'return_vocabulary_stats':True,
                    'return_tense_count':True,'return_tense_term_count':True,'return_tense_stats':True,'return_clause_count':True,
                    'return_clause_stats':True,'return_phrase_count':True,'return_final_levels':True}

            self.tense_level_dict = {('unbound modal verbs', 'fin.'):0,
                    ('do', 'inf.'):0,
                    ('do', 'imp.'):0,
                    ('do', 'ind. (present)'):0,
                    ('do', 'ind. (past)'):0.5,
                    ('be doing', 'ind. (present)'):0.5,
                    ('have done', 'ind. (present)'):1,
                    ('be doing', 'ind. (past)'):1.5,
                    ('be done', 'inf.'):2,
                    ('be done', 'ind. (present)'):2,
                    ('be done', 'ind. (past)'):2,
                    ('have done', 'ind. (past)'):2,
                    ('have been doing', 'ind. (present)'):2.5,
                    ('have been done', 'ind. (present)'):2.5,
                    ('be being done', 'ind. (present)'):2.5,
                    ('do', 'ger.'):1,
                    ('do', 'part. (past)'):2,
                    ('be done', 'ger.'):2.5,
                    ('be doing', 'inf.'):3,
                    ('have done', 'inf.'):3,
                    ('have been doing', 'ind. (past)'):3.5,
                    ('have been done', 'ind. (past)'):3.5,
                    ('be being done', 'ind. (past)'):3.5,
                    ('have been done', 'inf.'):4,
                    ('have done', 'ger.'):4,
                    ('have been doing', 'inf.'):4.5,
                    ('have been doing', 'ger.'):5,
                    ('have been done', 'ger.'):5}

            self.tense_name_dict = {('unbound modal verbs', 'fin.'):'Unbound modal verb',
                    ('do', 'inf.'):'Infinitive',
                    ('do', 'imp.'):'Imperative',
                    ('do', 'ind. (present)'):'Present simple',
                    ('do', 'ind. (past)'):'Past simple',
                    ('be doing', 'ind. (present)'):'Present continuous',
                    ('have done', 'ind. (present)'):'Present perfect',
                    ('be doing', 'ind. (past)'):'Past continuous',
                    ('be done', 'inf.'):'Infinitive passive',
                    ('be done', 'ind. (present)'):'Present simple passive',
                    ('be done', 'ind. (past)'):'Past simple passive',
                    ('have done', 'ind. (past)'):'Past perfect',
                    ('have been doing', 'ind. (present)'):'Present perfect continuous',
                    ('have been done', 'ind. (present)'):'Present perfect passive',
                    ('be being done', 'ind. (present)'):'Present continuous passive',
                    ('do', 'ger.'):'Gerund simple',
                    ('do', 'part. (past)'):'Past participle simple',
                    ('be done', 'ger.'):'Gerund perfect',
                    ('be doing', 'inf.'):'Infinitive continuous',
                    ('have done', 'inf.'):'Infinitive perfect',
                    ('have been doing', 'ind. (past)'):'Past perfect continuous',
                    ('have been done', 'ind. (past)'):'Past perfect passive',
                    ('be being done', 'ind. (past)'):'Past continuous passive',
                    ('have been done', 'inf.'):'Infinitive perfect passive',
                    ('have done', 'ger.'):'Gerund perfect passive',
                    ('have been doing', 'inf.'):'Infinitive perfect continuous',
                    ('have been doing', 'ger.'):'Gerund perfect continuous',
                    ('have been done', 'ger.'):'Gerund perfect passive'}

            #self.sentences = None
            #self.wordlists = None
            #self.stats_dict = None
            #self.tense_count = None
            #self.tense_stats = None
            #self.clause_count = None
            #self.clause_stats = None
            #self.final_levels = None
            self.result = None
            self.simplification = None

        def get_word_cefr(self,word_lemma,word_orth="",cefr_w_pos_prim=cefr_w_pos_min_prim,cefr_wo_pos_prim=cefr_wo_pos_min_prim):
            if word_orth=="":
                word_orth = word_lemma
            level = cefr_w_pos_prim.get(word_lemma,6)
            if level == 6:
                level = cefr_w_pos.get(word_lemma,6)
                if level == 6:
                    level = max(2,cefr_w_pos_sup.get(word_lemma,6))
                    if level == 6:
                        level = cefr_w_pos_prim.get(word_orth,6)
                        if level == 6:
                            level = cefr_w_pos.get(word_orth,6)
                            if level == 6:
                                level = max(2,cefr_w_pos_sup.get(word_orth,6))
                                if level == 6:
                                    level = cefr_wo_pos_prim.get(word_lemma[0],6)
                                    if level == 6:
                                        level = cefr_wo_pos.get(word_lemma[0],6)
                                        if level == 6:
                                            level = max(2,cefr_wo_pos_sup.get(word_orth[0],6))
            return level

        def get_phrase(self, phrase, phrase_pos, sentence, sentence_start_index, followed_by, window_size=3):
            
            confidence = length = len(phrase)
            
            sm = edit_distance.SequenceMatcher(a=phrase, b=sentence, action_function=edit_distance.highest_match_action)
            opcodes = sm.get_opcodes()
            
            filter_ = []
            for opcode in opcodes:
                if opcode[0] == 'replace':
                    if not (phrase_pos[opcode[1]]==self.shared_object.doc[opcode[3]+sentence_start_index].pos_ and phrase_pos[opcode[1]] in set(['PRON','DET'])):
                        return None, 0
                    else:
                        filter_.append(True)
                    if any([x in set(['oneself','yourself']) for x in phrase[opcode[1]]]) and any([x.endswith('self') or x.endswith('selves') for x in set(sentence[opcode[3]])]):
                        confidence += 0.5
                    confidence -= 1
                elif opcode[0] == 'equal':
                    filter_.append(True)
                    pos_original = phrase_pos[opcode[1]]
                    pos_in_sent = self.shared_object.doc[opcode[3]+sentence_start_index].pos_
                    if pos_original!=pos_in_sent and not (pos_original=='ADP' and pos_in_sent=='ADV' or 
                                                          pos_original=='ADV' and pos_in_sent=='ADP' or 
                                                          pos_original=="PROPN" or 
                                                          pos_in_sent=="PROPN"):
                        confidence -= 1

                    #if pos_original==pos_in_sent=="VERB" and len(set(sentence[opcode[3]])-set(phrase[opcode[1]]))>0:
                    #    if opcode[1]==0 and len(set(phrase[opcode[1]]))-len(set(sentence[opcode[3]]))>0 or opcode[1]>=1 and opcode[3]>1 and 'be' in phrase[opcode[1]-1]+sentence[opcode[3]-1]:
                    #        confidence -= 1
                else:
                    filter_.append(False)
            
            
            matching_blocks = np.array(opcodes)[filter_][:,[1,3]].astype(int)
            
            operations = np.array(opcodes).T[0]
            operations_count = Counter(operations)
            if ['outa'] in phrase:
                print(opcodes)
            if operations_count['equal']+operations_count['replace']!=length or operations_count['equal']<=length/2 or 'delete' in operations_count:
                return None, 0
            
            # 1: only one
            # 2: , or none
            # 3: something noun-like
            # 4: something not VERB
            for i in range(1,len(matching_blocks)):
                n_insertions = matching_blocks[i][1]-matching_blocks[i-1][1]-1
                if followed_by[i-1]=="0":
                    if opcodes[matching_blocks[i][1]-1][0] not in set(['equal','replace']):
                        return None, 0
                else:
                    if self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index].pos_=="VERB":
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.tag_ == "RP":
                                return None, 0
                    
                    if followed_by[i-1]=="1":
                        if n_insertions!=1:
                            confidence -= abs(1-n_insertions)
                    elif followed_by[i-1]==",":
                        if n_insertions>1:
                            confidence -= abs(1-n_insertions)
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i][1]-1+sentence_start_index].pos_ != 'PUNCT':
                            return None, 0
                    elif followed_by[i-1] in set(['p','t']):
                        person = False
                        thing = True
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.pos_=="PRON":
                                if x.orth_.lower() == 'them':
                                    person = True
                                    break
                                elif x.orth_.lower() in set(['me','you','him','her','we','everyone','everybody','someone','somebody','anyone','anybody','one','nobody']) or x.orth_.lower().endswith('self'):
                                    thing = False
                                    person = True
                                    break
                            elif x.pos_=='NOUN':
                                if len(set([x.lemma_,x.orth_]).intersection(people_list))>0:
                                    thing = False
                                    person = True
                                    break
                            elif x.head.pos_=='NOUN' and matching_blocks[i-1][1]+sentence_start_index+1<x.head.i<matching_blocks[i][1]+sentence_start_index:
                                if len(set([x.head.lemma_,x.head.orth_]).intersection(people_list))>0:
                                    thing = False
                                    person = True
                                    break
                        if followed_by[i-1]=='p' and not person:
                            confidence -= 1
                        elif followed_by[i-1]=='t' and not thing:
                            confidence -= 0.5
                    elif followed_by[i-1]=="a":
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.pos_ in set(['VERB','AUX']):
                                return None, 0
                        if n_insertions==0:
                            confidence -= 1
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i-1][1]+1+sentence_start_index].pos_ == 'DET':
                            return None, 0
                    elif followed_by[i-1]=="s":
                        is_possessive = False
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.pos_ in set(['VERB','AUX']):
                                return None, 0
                            elif x.lemma_ in set(["'s", "'", "my", "your", "his", "her", "our", "their"]):
                                is_possessive = True
                                break
                        if not is_possessive:
                            confidence -= 1
                    elif followed_by[i-1]=="n":
                        if self.shared_object.doc[matching_blocks[i-1][1]+1+sentence_start_index].pos_ != 'NUM':
                            confidence -= 1

                if n_insertions==0 and followed_by[i-1] in set(['a','p','t']) and self.shared_object.doc[matching_blocks[i][1]-1+sentence_start_index].tag_!='VBN':
                    confidence -= 1
                if n_insertions>window_size:
                    confidence -= n_insertions-window_size
            
            if confidence<=0:
                return None, 0
            
            span = list(np.array(matching_blocks)[:,1])

            phrase_start_index = span[0]+sentence_start_index
            phrase_end_index = span[-1]+sentence_start_index
            followed_by_something = phrase_end_index<self.shared_object.doc[phrase_end_index].sent[-1].i and self.shared_object.doc[phrase_end_index+1].lemma_.isalnum() and not (self.shared_object.doc[phrase_end_index+1].tag_=='CC' and set(phrase_pos).intersection(set(['VERB','AUX'])))
            if followed_by[-1]=='0':
                if followed_by_something:
                    confidence -= 1
            else:
                if followed_by[-1]!='a' and not followed_by_something and self.shared_object.doc[phrase_start_index].tag_!='VBN' and self.shared_object.doc[max(0,phrase_start_index-1)].lemma_!='to':
                    confidence -= 1
                if followed_by[-1] in set(['p','t']):
                    followed_by_person = False
                    followed_by_thing = True
                    if phrase_end_index<self.shared_object.doc[phrase_end_index].sent[-1].i:
                        x = self.shared_object.doc[phrase_end_index+1]
                        if x.pos_=="PRON":
                            if x.orth_.lower()=='them':
                                followed_by_person = True
                            elif x.orth_.lower() in set(['me','you','him','her','we','everyone','everybody','someone','somebody','anyone','anybody','one','nobody']) or x.orth_.lower().endswith('self'):
                                followed_by_thing = False
                                followed_by_person = True
                        elif x.pos_=='NOUN':
                            if len(set([x.lemma_,x.orth_]).intersection(people_list))>0:
                                followed_by_thing = False
                                followed_by_person = True
                        elif x.head.pos_=='NOUN' and phrase_end_index<x.head.i:
                            if len(set([x.head.lemma_,x.head.orth_]).intersection(people_list))>0:
                                followed_by_thing = False
                                followed_by_person = True
                    if followed_by[-1]=='p' and not followed_by_person:
                        confidence -= 1
                    elif followed_by[-1]=='t' and not followed_by_thing:
                        confidence -= 0.5
                elif followed_by[-1]=='n':
                    if phrase_end_index<len(self.shared_object.doc)-1 and self.shared_object.doc[phrase_end_index+1].pos_!='NUM':
                        confidence -= 1

            if confidence<=0:
                return None, 0

            return span, confidence/len(phrase)

        def get_sentence_parts(self, x, followed_by, window_size=3):
            followed_by = followed_by.split('_')[0]
            phrase_length = len(followed_by)+1+followed_by.count('1')+followed_by.count('2')+(followed_by.count('3')+followed_by.count('4'))*window_size
            sentence_parts = []
            start_index = max(x.sent[0].i, x.i-phrase_length)
            end_index = min(x.i+phrase_length, x.sent[-1].i)
            for i in range(start_index,end_index+1):
                if self.shared_object.doc[i].lemma_=='not':
                    sentence_parts.append(['not','never','hardly','barely'])
                elif (self.shared_object.doc[i].lemma_.endswith('self') or self.shared_object.doc[i].lemma_.endswith('selves')) and self.shared_object.doc[i].lemma_!='self':
                    sentence_parts.append([self.shared_object.doc[i].lemma_, self.shared_object.doc[i].orth_.lower(), 'oneself'])
                else:
                    sentence_parts.append([self.shared_object.doc[i].lemma_, self.shared_object.doc[i].orth_.lower()])
            return sentence_parts, start_index

        def get_verb_form(self,x):
            tense = []
            id_range = []

            if x.orth_.lower() == 'being' and str(x.morph)=='VerbForm=Ger' and self.shared_object.doc[min(x.i+1,len(self.shared_object.doc)-1)].pos_!='VERB':
                tense.append('being')
            elif x.head == x or x.dep_ == 'ROOT' or x.dep_ not in set(['aux','auxpass']):
                tense, id_range = self.get_aux(x)
                if len(tense) == 0 and x.dep_=='conj':
                    first_verb = x.head
                    while first_verb.dep_=='conj':
                        first_verb = first_verb.head
                    tense, id_range = self.get_aux(first_verb)

                if 'Aspect=Perf' in x.morph or (self.shared_object.doc[max(0,x.i-1)].lemma_=='have' and self.shared_object.doc[max(0,x.i-1)].pos_=='AUX' or self.shared_object.doc[max(0,x.i-2)].lemma_ == 'have' and self.shared_object.doc[max(0,x.i-2)].pos_=='AUX') and 'VerbForm=Fin' in x.morph and 'Tense=Past' in x.morph:
                    tense.append('done')
                elif ('Mood=Ind' in x.morph or 'VerbForm=Ger' in x.morph or 'Tense=Past' in x.morph and 'VerbForm=Part' in x.morph) and x.lemma_ in set(['be','have']):
                    if x.orth_.startswith("'") or x.orth_.startswith("’"):
                        tense.append(self.expand_contraction(x))
                    else:
                        tense.append(x.orth_.lower())
                elif 'VerbForm=Fin' in x.morph:
                    if self.shared_object.doc[max(0,x.i-1)].lemma_=='to' or self.shared_object.doc[max(0,x.i-2)].lemma_=='to':
                        tense.append('do')
                    elif 'Tense=Past' in x.morph:
                        tense.append('did')
                    elif 'Tense=Pres' in x.morph:
                        if 'Person=3' in x.morph:
                            tense.append('does')
                        else:
                            tense.append('do')
                    else:
                        tense.append(x.lemma_)
                elif 'Aspect=Prog' in x.morph:
                    tense.append('doing')
                elif 'VerbType=Mod' in x.morph:
                    tense.append(x.lemma_)
                elif 'VerbForm=Inf' in x.morph:
                    tense.append('do')
                
            if len(tense)==0:
                return None, None
            else:
                if len(set(tense))!=len(tense):
                    new_tense = [tense[0]]
                    for t in np.array(tense)[1:]:
                        if t != new_tense[-1] or t=='do':
                            new_tense.append(t)
                    tense = new_tense
                      
                tense = ' '.join(tense).replace("have been","have bEEn").replace("has been","has bEEn").replace("had been","had bEEn").replace('got ta','gotta').replace('gon na','gonna').replace(
                    "have be","have been").replace("has be","has been").replace("had be","had been").lower()
                
                if tense.endswith('ca'):
                    tense = tense.replace('ca','can')
                elif tense.endswith('wo'):
                    tense = tense.replace('wo','will')
                elif tense.endswith('sha'):
                    tense = tense.replace('sha','shall')
                elif tense.endswith('ai'):
                    tense = tense.replace('ai',"ain't")
                    
                tense = tense.replace('ca ','can ').replace('wo ','will ').replace('sha ','shall ').replace('ai ',"ain't ")
                
                # Fix some form errors with "have"
                if tense in set(['has','have','had']) and (
                    (self.shared_object.doc[min(x.i+1,len(self.shared_object.doc)-1)].lemma_ == 'to' and self.shared_object.doc[min(x.i+2,len(self.shared_object.doc)-1)].pos_ in set(['AUX','VERB'])) or (
                        x.pos_!='AUX' and any([child.dep_=='dobj' for child in x.children if x.i<child.i]))):
                    tense = {'has':'does','have':'do','had':'did'}[tense]
                elif tense == 'had do':
                    if self.shared_object.doc[max(0,x.i-1)].orth_.lower()!='better':
                        tense = 'had done'
                    else:
                        tense = 'had better do'
                elif tense in set(['have do','has do','having do','having been do']):
                    tense = ' '.join(tense.split(' ')[:-1]+['done'])
                elif tense == 'has doing':
                    tense = 'is doing'
                id_range.append(x.i)
                return tense, list(set(id_range))

        def expand_contraction(self,x):
            t = x.orth_.lower()
            if x.orth_.lower().endswith('ve'):
                t = 'have'
            elif x.orth_.lower().endswith('re'):
                t = 'are'
            elif x.orth_.lower().endswith('m'):
                t = 'am'
            elif x.orth_.lower().endswith('d'):
                if x.lemma_ == 'have':
                    t = 'had'
                else:
                    t = 'would'
            elif x.orth_.lower().endswith('ll'):
                t = 'will'
            elif x.orth_.lower().endswith('s'):
                if x.lemma_ == 'be':
                    t = 'is'
                    if any([x.i!=x.head.i and child.i>x.head.i and child.pos_ in set(['NOUN','PRON','PROPN']) for child in x.head.children]):
                        t = 'has'
                else:
                    t = 'has'
            return t
        
        def get_aux(self,x):
            tense = []
            id_range = []
            if x.dep_=='xcomp' and list(x.children)[0].lemma_!='to' and list(x.children)[0].pos_!='PART':
                return tense, id_range
            for child in x.children:
                if child.dep_ in set(['aux','auxpass']) and child.i<x.i:# and child.i<x_i
                    if child.orth_ in set(['…','...',';','-','—','–',':']):
                        tense = []
                        continue
                    if child.orth_.startswith("'") or child.orth_.startswith("’"):
                        tense.append(self.expand_contraction(child))
                        id_range.append(child.i)
                    else:
                        tense.append(child.orth_.lower())
                        id_range.append(child.i)
            return tense, id_range

        def classify_tense(self,form,x):
            tense2 = None
            tense1 = None
            
            if form.startswith("ain't"):
                tense2 = 'contextual'
            elif form == 'had better do':
                tense2 = 'inf.'
            elif sum([form.startswith(y) for y in set(['did','was','were','had','got'])])>0:
                tense2 = 'ind. (past)'
            elif form == 'do do' and x.i>=3 and self.shared_object.doc[x.i-1].lemma_=='you' and self.shared_object.doc[x.i-2].lemma_=='not' and self.shared_object.doc[x.i-3].lemma_=='do' and all([child.orth_!='?' for child in x.children if x.i<child.i]):
                tense2 = 'imp.'
            elif sum([form.startswith(y) for y in set(['do ','does','has','have','am','is','are','gets'])])>0:
                tense2 = 'ind. (present)'
            elif form=='do':
                first = x
                while first.dep_ == 'conj':
                    first = first.head
                if all([child.dep_!='nsubj' for child in first.children if child.i<first.i]) and str(x.morph) == 'VerbForm=Inf':
                    tense2 = 'imp.'
                elif "VerbForm=Fin" in str(x.morph) and first.head.lemma_!='let':
                    tense2 = 'ind. (present)'
                else:
                    tense2 = 'inf.'
            elif form.endswith('do'):
                tense2 = 'inf.'
            elif form=='done':
                tense2 = 'part. (past)'
            elif form.split(' ')[0].endswith('ing'):
                tense2 = 'part. (present)'
                if x.dep_ == 'conj':
                    first = x.head
                    while first.dep_ == 'conj':
                        first = first.head
                else:
                    first = x
                if first.head.pos_ in set(['VERB','ADP']) and first.dep_ in set(['xcomp','pcomp']) or first.dep_=='ccomp' and first.head.pos_=='VERB' and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([first.head.i,first.i])) if i!=first.head.i]) or first.head.pos_=='AUX' and first.dep_=='csubj':
                    tense2 = 'ger.'
            elif str(x.morph) in set(['VerbForm=Fin','VerbType=Mod']):
                tense2 = 'fin.'
            else:
                tense2 = 'inf.'

            if form.endswith('done'):
                if tense2 == 'part. (past)':
                    tense1 = 'do'
                else:
                    if form.endswith('been done'):
                        tense1 = 'have been done'
                    elif form.endswith('being done'):
                        if form == 'being done':
                            tense1 = 'be done'
                        else:
                            tense1 = 'be being done'
                    elif form in set(['have done','has done','had done']):
                        tense1 = 'have done'
                    elif form.startswith('having'):
                        tense1 = 'have '+' '.join(form.split(' ')[1:])
                    else:
                        tense1 = 'be done'
                    
            elif form.endswith('been'):
                tense1 = 'have done'
                
            elif form.endswith('doing'):
                if form.endswith('been doing'):
                    tense1 = 'have been doing'
                else:
                    if tense2 in set(['part. (present)','ger.']):
                        tense1 = 'do'
                    elif form.startswith('having'):
                        tense1 = 'have '+' '.join(form.split(' ')[1:])
                    elif form.startswith('be ') or form.startswith('am ') or form.startswith('is ') or form.startswith('are ') or form.startswith('was ') or form.startswith('were ') or form.startswith("ain't "):
                        tense1 = 'be doing'
                
            elif form == 'being':
                tense1 = 'do'
                
            elif form in set(['has','had','had']):
                if self.shared_object.doc[min(x.i+1,len(self.shared_object.doc)-1)].pos_ == 'PUNCT':
                    tense1 = 'have done'
                else:
                    tense1 = 'do'
            elif tense2 == 'fin.':
                tense1 = 'unbound modal verbs'
            else:
                tense1 = 'do'
                
            if tense2.startswith('part'):
                tense1 = None
                tense2 = None
            return tense1,tense2

        def tense2level(self,form,tense1,tense2,x):
            tenses = self.tense_level_dict
                
            level = tenses.get((tense1,tense2),5)

            #if form in set(['do do','did do','does do']):
            #    if not any([y.lemma_ == 'not' for y in doc[x.head.i:x.i]]):
            #        level = 1
            if any([form.startswith(m) for m in set(['will','may','might','would','could','shall'])]):
                level = max(1,level)
            elif any([form.startswith(m) for m in set(['can','should','must'])]):
                level = max(0.5,level)
            elif tense1=='have done' and tense2=='ind. (present)' and x.lemma_.lower()=='get' and all([child.dep_ != 'dative' for child in x.children]) and any([child.dep_ == 'dobj' for child in x.children]):
                level = 0
                form = 'have got (POSSESS)'
            return level, form

        def convert_tense_name(self,form,tense1,tense2):
            name = self.tense_name_dict.get((tense1,tense2),"?")
            if tense2 == 'inf.':
                if form.startswith('will'):
                    if ' ' in name:
                        name = 'Future {} ({} with modal verb "will")'.format(' '.join(name.split(' ')[1:]),name)
                    else:
                        name = f'Future simple ({name} with modal verb "will")'
                elif form.startswith('to '):
                    name += ' with "to"'
                elif any([form.startswith(m) for m in set(['may','might','would','could','shall','can','should','must'])]):
                    name += ' with modal verb: {}'.format(form.split(' ')[0])
                    #name += ' with modal verb'
                elif len(form.split(' '))>=2:
                    name += ' with auxilary verb'
                else:
                    name += ' without "to"'
            elif form == 'have got (POSSESS)':
                name = 'Possession with "have got"'
            return name

        def get_subtree(self,members,new_members):
            if len(new_members) == 0:
                return members
            else:
                members += new_members
                children = []
                for i in new_members:
                    for child in self.shared_object.doc[i].children:
                        #if child.dep_ not in set(['relcl','ccomp','advcl','acl','csubj']) and not (child.dep_=='pcomp' and child.pos_ in set(['VERB','AUX'])):
                        if child.dep_ not in set(['relcl','ccomp','advcl','acl','csubj']):
                            children.append(child.i)
                return self.get_subtree(members,children)

        def get_clause(self,x):
            subtree = None
            clause = None
            clause_form = None
            level = None
            leftest_child = None
            children = []
            
            if x.dep_ == 'conj' and x.nbor(-1).dep_!='cc':
                first = x.head
                while first.dep_ == 'conj':
                    first = first.head
            else:
                first = x
            if (first.dep_ == 'advcl' or first.dep_ == 'acl') and x.pos_ in set(['VERB','AUX']):
                for child in x.children:
                    if child.dep_ not in set(['relcl','advcl','ccomp','acl','csubj']) and not (child.dep_=='conj' and self.shared_object.doc[max(0,child.i-1)].dep_!='cc'):
                        children.append(child.i)
                        if clause_form is None and child.i<x.i:
                            if child.tag_ == 'VBG':
                                clause_form = 'part. (present)'
                            elif child.tag_ == 'VBN':
                                clause_form = 'part. (past)'
                            elif child.tag_ in set(['WRB','WDT','WP','WP$']) or child.tag_=='IN' and len(list(child.children))==0:
                                clause_form = child.lemma_
                subtree = sorted(self.get_subtree([x.i],children))
                if clause_form is None:
                    try:
                        leftest_child = self.shared_object.doc[min([child.i for child in first.children if child.i<first.i])]
                    except:
                        pass
                    if leftest_child is not None and (leftest_child.tag_ in set(['WRB','WDT','WP','WP$']) or leftest_child.tag_=='IN' and len(list(leftest_child.children))==0):
                        clause_form = leftest_child.lemma_
                    elif self.shared_object.doc[subtree[0]].orth_.lower() in set(['had','should','were']) and self.shared_object.doc[subtree[0]].pos_=='AUX':
                        clause_form = self.shared_object.doc[subtree[0]].orth_.lower()
                    elif self.shared_object.doc[subtree[0]].tag_ in set(['WRB','WDT','WP','WP$']) or self.shared_object.doc[subtree[0]].tag_=='IN' and len(list(self.shared_object.doc[subtree[0]].children))==0:
                        clause_form = self.shared_object.doc[subtree[0]].lemma_
                    else:
                        if first.tag_ == 'VBG' and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([subtree[0],first.i]))]):
                            clause_form = 'part. (present)'
                        elif first.tag_ == 'VBN' and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([subtree[0],first.i]))]):
                            clause_form = 'part. (past)'
                        elif first.tag_ in set(['WRB']) or first.tag_=='IN' and len(list(first.children))==0:
                            clause_form = x.lemma_
                if clause_form is None and leftest_child is not None:
                    if leftest_child.tag_=='RB':
                        clause_form = leftest_child.lemma_
                    elif leftest_child.lemma_!='to':
                        clause_form = '(that)'
                
            elif first.dep_ == 'relcl' and 'TO' not in set([child.tag_ for child in x.children if child.i < x.i]):
                for child in x.children:
                    if child.dep_ not in set(['relcl','advcl','ccomp','acl','csubj']) and not (child.dep_=='conj' and self.shared_object.doc[max(0,child.i-1)].dep_!='cc'):
                        children.append(child.i)
                        if clause_form is None and child.tag_ in set(['WRB','WDT','WP','WP$']) and child.i<x.i:
                            clause_form = child.lemma_
                if clause_form is None:
                    for grandchild in sum([list(child.children) for child in x.children if child.i<x.i],[]):
                        if grandchild.tag_ in set(['WRB','WDT','WP','WP$']):
                            clause_form = grandchild.lemma_
                            break
                if clause_form is None:
                    clause_form = '(that)'
                subtree = self.get_subtree([x.i],children)
            elif first.dep_ in set(['ccomp','csubj']) and x.pos_ in set(['VERB','AUX']) and all([self.shared_object.doc[j].lemma_ not in set(['"',';']) for j in range(*sorted([x.i,first.head.i]))]):
                for child in x.children:
                    if child.dep_ not in set(['relcl','advcl','ccomp','acl','csubj']) and not (child.dep_=='conj' and self.shared_object.doc[max(0,child.i-1)].dep_!='cc'):
                        children.append(child.i)
                        if clause_form is None and child.i<x.i and child.tag_ in set(['WRB','WDT','WP','WP$','IN']) and child.pos_!='ADP' and not (first.dep_=='ccomp' and x.i<=first.head.i):
                            clause_form = child.lemma_
                subtree = sorted(self.get_subtree([x.i],children))
                if clause_form is None and not (first.dep_=='ccomp' and x.i<=first.head.i):
                    for grandchild in sum([list(child.children) for child in x.children if child.i<x.i and child.pos_ not in set(['VERB','AUX'])],[]):
                        if grandchild.tag_ in set(['WRB','WDT','WP','WP$']):
                            clause_form = grandchild.lemma_
                            break
                if clause_form is None:
                    if self.shared_object.doc[subtree[0]].tag_ in set(['WRB','WDT','WP','WP$']) and not (first.dep_=='ccomp' and x.i<=first.head.i):
                        clause_form = self.shared_object.doc[subtree[0]].lemma_
                    elif x.tag_=='VBN' and (x.head.pos_=='VERB' or x.head.pos_=='AUX' and x.i<x.head.i) and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([subtree[0],first.i]))]):
                        clause_form = 'part. (past)'
                    elif x.tag_=='VBG' and (x.head.pos_=='VERB' or x.head.pos_=='AUX' and x.i<x.head.i) and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([subtree[0],first.i]))]):
                        clause_form = None
                    elif x.tag_=='VB' and all(self.shared_object.doc[i].pos_!='AUX' for i in range(*sorted([first.head.i,first.i]))):
                        clause_form = None
                    elif first.head.lemma_!='let' and first.head.i<x.i:
                        clause_form = '(that)'
                if clause_form is not None and not (all(self.shared_object.doc[y].lemma_!=')' for y in range(*sorted([first.head.i,first.i]))) or any(self.shared_object.doc[y].lemma_=='(' for y in range(*sorted([first.head.i,first.i])))):
                    clause_form = None
                last_i = sorted(set(subtree))[-1]
                if first.dep_ == 'ccomp' and (self.shared_object.doc[min(last_i+1,len(self.shared_object.doc)-1)].lemma_=='so' or self.shared_object.doc[min(last_i+2,len(self.shared_object.doc)-1)].lemma_=='so' or self.shared_object.doc[min(last_i+3,len(self.shared_object.doc)-1)].lemma_=='so'):
                    clause_form = None
                    subtree = None
            elif first.dep_=='pcomp' and x.pos_ in set(['VERB','AUX']):
                for child in x.children:
                    if child.dep_ not in set(['relcl','advcl','ccomp','acl','csubj']) and not (child.dep_=='conj' and self.shared_object.doc[max(0,child.i-1)].dep_!='cc'):
                        children.append(child.i)
                        if clause_form is None and child.i<x.i and child.tag_ in set(['WRB','WDT','WP','WP$']):
                            clause_form = child.lemma_
                if clause_form is None:
                    for grandchild in sum([list(child.children) for child in x.children if child.i<x.i],[]):
                        if grandchild.tag_ in set(['WRB','WDT','WP','WP$']):
                            clause_form = grandchild.lemma_
                            break
                subtree = sorted(self.get_subtree([x.i],children))
                    
            elif first.dep_=='prep' and first.pos_=='VERB' and first.tag_=='VBN':
                subtree = self.get_subtree([x.i],[child.i for child in x.children])
                clause_form = 'part. (past)'
                
            #elif first.dep_ == 'parataxis':
            #    clause_form = '(parataxis)'
                
            if subtree is None or clause_form is None:
                if x.dep_=='conj' and x.pos_ in set(['VERB','AUX']):
                    temp_children = list(x.children)
                    if len(temp_children)>0 and temp_children[0].dep_ == 'mark':
                        clause_form = temp_children[0].lemma_
                        subtree = sorted(self.get_subtree([x.i],children))
                    else:
                        temp_siblings = []
                        has_subject = False
                        for sibling in x.head.children:
                            if sibling.i<x.i and sibling.pos_!= 'PUNCT' and sibling.pos_!= 'SPACE':
                                temp_siblings.append(sibling)
                        if len(temp_siblings)>0:
                            for i in range(temp_siblings[-1].i,x.i):
                                if self.shared_object.doc[i].dep_ == 'nsubj':
                                    has_subject = True
                            if has_subject and temp_siblings[-1].dep_ == 'cc' and temp_siblings[-1].i<x.i:
                                clause_form = temp_siblings[-1].lemma_
                                clause = 'cc'
                                subtree = sorted(self.get_subtree([x.i],children)+[temp_siblings[-1].i])
                elif x.pos_ == 'CONJ' and x.head.dep_ == 'ROOT' and x.i<x.head.i:
                    clause_form = x.lemma_
                    clause = 'cc'
                    subtree = list(range(x.i,x.head.i+1))

            if subtree is not None and clause_form is not None and len(subtree)>0:
                if clause_form == '(parenthesis)':
                    clause = 'parenthesis'
                elif clause != 'cc':
                    clause = first.dep_
                if clause in set(['ccomp','csubj','pcomp']):
                    clause = 'ncl'
                elif clause == 'prep':
                    clause = 'advcl'
                elif clause in set(['relcl','acl']) and x.head.orth_.lower()=='idea' and x.head.head.lemma_ in set(['have','get']):
                    clause = 'ncl'
                subtree = sorted(set(subtree))
                level = self.clause2level(clause_form,clause,subtree,first,leftest_child)
            return clause_form, clause, subtree, level

        def clause2level(self,clause_form,clause,subtree,first,leftest_child):
            level = 1
            if clause == 'relcl':
                if any([child.pos_=='ADP' for child in first.children if child.i<first.i]):
                    level = 3
                elif clause_form in set(['whose','where','when','why','how','whether','what']) or self.shared_object.doc[subtree[-1]].pos_=='ADP' and len(list(self.shared_object.doc[subtree[-1]].children))==0:
                    level = 2
                else:
                    level = 1
                    
            elif clause == 'ncl':
                if first.dep_ == 'csubj':
                    level = 4
                elif clause_form.endswith('ever'):
                    level = 3
                elif clause_form.startswith('part.'):
                    level = 3
                elif clause_form in set(['whose','where','when','why','how','whether','what','who','which','if']):
                    level = 3
                elif clause_form in set(['that','(that)']):
                    level = 2
                    
            elif clause == 'advcl':
                if clause_form.endswith('ever'):
                    level = 4
                elif clause_form.startswith('part.'):
                    level = 2
                elif (leftest_child is not None and leftest_child.tag_=='IN' or self.shared_object.doc[min(subtree)].tag_=='IN') and first.tag_ in set(['VBG','VBN']) and all([self.shared_object.doc[i].tag_ not in set(['VBZ','VBD','VBP','VB']) for i in range(*sorted([min(subtree),first.i]))]):
                    level = 4
                elif clause_form in set(['that','(that)','when']):
                    level = 1
                elif clause_form in set(['because','so','but']):
                    level = 0
                elif clause_form in set(['whether','since','as']):
                    level = 2
                elif clause_form in set(['where','how','what']):
                    level = 4
                elif clause_form in set(['had','should','were']):
                    level = 5
                
            elif clause == 'prep':
                if clause_form=='part. (past)':
                    level = 4
                    
            elif clause == 'acl':
                level = 2
                
            elif clause == 'parenthesis':
                level = 2

            elif clause == 'cc':
                if clause_form == 'yet':
                    level = 2
                else:
                    level = 0
                
            return level

        def process_input(self,reference_word,reference_pos,reference_CEFR,word,pos,max_length=38):
            letter2id = {x:i+7 for i,x in enumerate(list('abcdefghijklmnopqrstuvwxyz'))}
            pos2id = {'NOUN': 33, 'ADJ': 34, 'VERB': 35, 'ADV': 36}
            x =  [letter2id[x] for x in reference_word] + [pos2id[reference_pos]] + [reference_CEFR+1] + [letter2id[x] for x in word] + [pos2id[pos]]
            return np.pad(x,(0,max_length-len(x)))

        def stem_prefix(self,word):
            prefixes = ["anti","auto","de","dis","down","extra","hyper","il","im","in","ir",
                                "inter","mega","mid","mis","non","over","out","post","pre","pro","re",
                                "semi","sub","super","tele","trans","ultra","un","up"]
            for prefix in sorted(prefixes, key=len, reverse=True):
                if word.startswith(prefix):
                    word = word[len(prefix):]
            return word

        def stem_suffix(self,word):
            suffixes = ['less']
            for suffix in sorted(suffixes, key=len, reverse=True):
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
            return word

        def get_stems(self,word):
            stems = set(sum([[stemmer.stem(self.stem_prefix(word)),stemmer.stem(word)] for stemmer in stemmers],[]))
            stems = stems.union(set([self.stem_suffix(stem) for stem in stems]))
            return stems

        def predict_cefr(self,word,pos):
            levels = []
            word = str(word).replace('-','')
            stems = self.get_stems(word)
            for stem in stems:
                for _,row in df_reference_words[df_reference_words['level_0']==stem].iterrows():
                    try:
                        levels.append(np.argmax(cefr_word_model.predict(self.process_input(*row.values,word,pos).reshape(1,-1))))
                    except:
                        pass
            if len(levels)==0:
                for stem in stems:
                    for _,row in df_reference_words_sup[df_reference_words_sup['level_0']==stem].iterrows():
                        try:
                            levels.append(np.argmax(cefr_word_model.predict(self.process_input(*row.values,word,pos).reshape(1,-1))))
                        except:
                            pass

            if len(levels)==0:
                return 6
            else:
                return max(levels)

        def get_tense_tips_data(self, tense_count, general_level):
            grammar_count = {}
            grammar_level = {}
            infinitive_temp = {}
            for v in tense_count.values():
                for i in range(len(v['tense_term'])):
                    name = v['tense_term'][i]
                    if name.startswith('Infinitive') and 'with modal verb' in name:
                        name_with_level = str(v['level'][i])+'_'+' '.join(name.split(' ')[:-1])
                        if name_with_level in infinitive_temp:
                            infinitive_temp[name_with_level].append(name.split(' ')[-1])
                        else:
                            infinitive_temp[name_with_level] = [name.split(' ')[-1]]
                    else:
                        if name in grammar_count:
                            grammar_count[name] += v['size'][i]
                        else:
                            grammar_count[name] = v['size'][i]
                        grammar_level[name] = v['level'][i]
            for name_with_level,modal_verbs in infinitive_temp.items():
                name = name_with_level.split('_')[-1] + ' {}'.format(', '.join(set(modal_verbs)))
                grammar_level[name] = float(name_with_level.split('_')[0])
                grammar_count[name] = len(modal_verbs)

            df_grammar = pd.concat([pd.Series(grammar_level,name='level'),pd.Series(grammar_count,name='count')],axis=1)
            df_grammar['level_diff'] = round(general_level-df_grammar['level'],1)
            df_grammar['ratio'] = np.round(df_grammar['count']/sum(df_grammar['count']),2)
            df_grammar['tense_term'] = df_grammar.index
            return df_grammar.sort_values(['level_diff','count'],ascending=[True,False])[['tense_term','level','level_diff','count','ratio']].to_dict('list')
            

        def process(self):
            dfs = {}
            rows = []
            for sent in self.shared_object.doc.sents:
                n_sent = len(dfs)
                for count_token, x in enumerate(sent):
                    #if x.pos_ == 'SPACE':
                    #    continue

                    ######################
                    # Remove extra spaces
                    #####################
                    the_orth = x.orth_
                    is_white_space = bool(x.whitespace_)
                    if x.orth_ == '"':
                        is_white_space = False
                        
                    if count_token+1 < len(sent) and x.orth_ == ' ' and sent[count_token+1].orth_ in set(['[', '(', '"']):
                        the_orth = ''
                        
                    if count_token+1 < len(sent) and sent[count_token+1].orth_ in set(['[', '(', '"']):
                        is_white_space = False

                    if the_orth == '' and not is_white_space:
                        continue
                    if the_orth.strip(' ') == '':
                        if len(rows)==0:
                            continue
                        elif not rows[-1]['whitespace']:
                            rows[-1]['whitespace'] = True
                            continue
                    elif '\n' in the_orth:
                        the_orth = the_orth.strip(' ')

                    ###################################

                    if x.pos_ == 'PUNCT' or x.pos_ == 'SPACE':
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':-2,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                    'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True})
                    elif not bool(re.match(".*[A-Za-z]+",x.lemma_)):
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':'PUNCT','CEFR':-2,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                    'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True})
                    else:
                        if x.pos_ == 'INTJ' and self.__settings['intj_as_lowest']==True:
                            rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':-1,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                        'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                        'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                        'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True})
                            continue
                        elif x.pos_ == 'PROPN':
                            if self.__settings['propn_as_lowest']==True:
                                rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':-1,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                            'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                            'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                            'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True})
                                continue
                            else:
                                x.lemma_ = lemmatizer.lemmatize(x.lemma_.lower())
                            
                        #x = fine_lemmatize(x,self.shared_object.doc,nlp)

                        tense_level = None
                        form = None
                        tense_span = None
                        tense1 = None
                        tense2 = None
                        tense_term = None

                        # Verb forms
                        try:
                            if x.pos_ in set(['VERB','AUX']):
                                form, tense_span = self.get_verb_form(x)
                        except:
                            pass
                        if form is not None:
                            tense1, tense2 = self.classify_tense(form,x)
                            if tense1 is not None and tense2 is not None:
                                tense_level, form = self.tense2level(form,tense1,tense2,x)
                                tense_term = self.convert_tense_name(form,tense1,tense2)
                                
                        # Clauses
                        clause_form, clause, clause_span, clause_level = self.get_clause(x)
                        if x.orth_.lower() in stopwords.words('english'):
                            word_lemma = tuple([x.lemma_,'STOP'])
                            word_orth = tuple([x.orth_.lower(),'STOP'])
                        else:
                            word_lemma = tuple([x.lemma_,x.pos_])
                            word_orth = tuple([x.orth_.lower(),x.pos_])

                        # Vocabulary
                        if self.__settings['keep_min']:
                            cefr_w_pos_prim = cefr_w_pos_min_prim
                            cefr_wo_pos_prim = cefr_wo_pos_min_prim
                        else:
                            cefr_w_pos_prim = cefr_w_pos_mean_prim
                            cefr_wo_pos_prim = cefr_wo_pos_mean_prim

                        level = self.get_word_cefr(word_lemma,word_orth,cefr_w_pos_prim,cefr_wo_pos_prim)

                        if level == 6:
                            if x.lemma_.endswith('1st') or x.lemma_.endswith('2nd') or x.lemma_.endswith('3rd') or bool(re.match("[0-9]+th$",x.lemma_)):
                                level = 0
                            elif x.pos_ == 'NUM':
                                if bool(re.match("[A-Za-z]+",x.lemma_)):
                                    level = 0
                                else:
                                    level = -1
                            elif len(re.findall("[A-Za-z]{1}",x.lemma_))==1:
                                level = 0
                            else:
                                level = max(2,self.predict_cefr(x.lemma_,x.pos_))

                        # Phrases
                        phrase = None
                        phrase_span = None
                        max_confidence = 0
                        ambiguous = True
                        if x.pos_ not in set(["DET","PART"]) and x.lemma_ in df_phrases['word'].values:
                            #max_phrase_length = 0
                            max_clean_length = 0

                            df_phrases_temp = df_phrases[df_phrases['word']==x.lemma_]
                            sentence_parts = []
                            for _, row in df_phrases_temp.iterrows():

                                if phrase is not None and phrase.startswith(row['original']) and max_confidence==1:
                                    continue

                                phrase_parts = row['phrase_parts']

                                #if phrase_length > max_phrase_length:
                                #    sentence_parts, start_index = self.get_sentence_parts(x,phrase_length)
                                #    max_phrase_length = phrase_length

                                sentence_parts, start_index = self.get_sentence_parts(x,row['followed_by'])

                                if len(phrase_parts)>len(sentence_parts) or len(set(sum(sentence_parts,[])).intersection(set(row['lemma'].split(' '))))<len(set(sum(phrase_parts,[]))):
                                    continue
                                phrase_span_temp, confidence_temp = self.get_phrase(phrase_parts, row['pos'].split(' '), sentence_parts, start_index, row['followed_by'])

                                if phrase_span_temp is not None and confidence_temp>0 and (confidence_temp>max_confidence or confidence_temp==max_confidence and len(phrase_parts)>max_clean_length):
                                    phrase_span = list(np.array(phrase_span_temp) + start_index)
                                    phrase = row['original']
                                    max_clean_length = len(phrase_parts)
                                    max_confidence = confidence_temp
                                    ambiguous = row['ambiguous']
                                    #ambiguous=False
                                    
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':level,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':form,'tense1':tense1,'tense2':tense2,'tense_term':tense_term,'CEFR_tense':tense_level,'tense_span':tense_span,
                                    'clause_form':clause_form,'clause':clause,'CEFR_clause':clause_level,'clause_span':clause_span,
                                    'phrase':phrase, 'phrase_span':phrase_span,'phrase_confidence':max_confidence,'phrase_ambiguous':ambiguous})

                df_lemma = pd.DataFrame(rows)
                if len(rows)>0 and len(df_lemma[df_lemma['CEFR']>=-1])>0:
                    dfs[n_sent] = df_lemma
                    rows = []

            if len(dfs)>0:
                df_lemma = pd.concat(dfs.values())
            else:
                df_lemma = pd.DataFrame([],columns=['id','word','lemma','pos','CEFR','whitespace','sentence_id',
                                    'form','tense1','tense2','tense_term','CEFR_tense','tense_span',
                                    'clause_form','clause','CEFR_clause','clause_span',
                                    'phrase', 'phrase_span','phrase_confidence','phrase_ambiguous'])
            n_words = len(df_lemma[(df_lemma['pos']!='PUNCT')&(df_lemma['pos']!='SPACE')])
            
            n_clausal = 0
            n_clauses = 0
            clause_levels = []
            sentences = {}

            dfs_phrase_count = []

            for sentence_id, df in dfs.items():
                #clause_level = self.sentence_clause_level(df['CEFR_clause'].dropna().values)
                #clause_levels.append((np.exp(clause_level)-0.9)*len(df[df['pos']!='PUNCT']))
                total_span = max(1,len(set(sum(df['clause_span'].dropna().values,[]))))
                level_by_clause = max(max(df['CEFR_clause'].fillna(0)),sum(df['CEFR_clause'].fillna(0).values*df['clause_span'].fillna('').apply(len).values)/total_span)
                level_by_length = min(max(0,1.1**len(df[(df['pos']!='PUNCT')&(df['pos']!='SPACE')])-1.5),7)
                clause_level = min(np.nanmean([level_by_length,level_by_clause]),6)
                clause_levels.append(clause_level)

                if self.__settings['return_phrase_count']:
                    df2 = df[['phrase','phrase_span','phrase_confidence','phrase_ambiguous','sentence_id']].dropna()
                    if len(df2)>0:
                        filter_ = []
                        spans = df2['phrase_span'].values
                        for i in range(len(spans)):
                            unique = True
                            for j in range(len(spans)):
                                if i!=j and spans[i][0]>=spans[j][0] and spans[i][-1]<=spans[j][-1]:
                                    unique = False
                                    break
                            filter_.append(unique)
                        df2 = df2[filter_]
                        dfs_phrase_count.append(df2)
                        phrase_dict = df2[['phrase','phrase_span','phrase_confidence','phrase_ambiguous']].to_dict(orient='list')
                    else:
                        phrase_dict = {'phrase':[],'phrase_span':[],'phrase_confidence':[],'phrase_ambiguous':[]}

                if self.__settings['return_sentences']:
                    lemma_dict = df[['id','word','lemma','pos','whitespace','CEFR']].to_dict(orient='list')
                    df2 = df[['form','tense1','tense2','CEFR_tense','tense_span']].dropna()
                    level_dict = {'CEFR_vocabulary':6,'CEFR_tense':5}
                    if len(df2)>0:
                        tense_dict = df2[['form','tense1','tense2','CEFR_tense','tense_span']].to_dict(orient='list')
                        level_dict['CEFR_tense'] = max(tense_dict['CEFR_tense'])
                    else:
                        tense_dict = {'form':[],'tense1':[],'tense2':[],'CEFR_tense':[],'tense_span':[]}
                        level_dict['CEFR_tense'] = 0

                    df2 = df[['clause_form','clause','clause_span']].dropna()
                    if len(df2)>0:
                        n_clausal += 1
                        n_clauses += len(df2)

                        df2['span_string'] = df2['clause_span'].astype(str)
                        clause_dict = df2.drop_duplicates(['span_string','clause'])[['clause_form','clause','clause_span']].to_dict(orient='list')
                    else:
                        clause_dict = {'clause_form':[],'clause':[],'clause_span':[]}

                    _,cumsum_series = self.sum_cumsum(df[df['CEFR']>=-1])

                    level_dict['CEFR_vocabulary'] = self.ninety_five(cumsum_series)
                    level_dict['CEFR_clause'] = round(clause_level,1)

                    if self.__settings['return_phrase_count']:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict,**phrase_dict}
                    else:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict}
                
            if self.__settings['return_wordlists']:
                wordlists = {}
                for CEFR,group in df_lemma[df_lemma['CEFR']>=-1].groupby('CEFR'):
                    df = group[['lemma','pos']]
                    wordlists[CEFR] = df.groupby(df.columns.tolist(),as_index=False).size().sort_values(['size','lemma','pos'],ascending=[False,True,True]).to_dict(orient='list')
                
            if self.__settings['return_tense_count'] or self.__settings['return_tense_stats']:
                tense_count = {}
                for tense,group in df_lemma[~pd.isnull(df_lemma['form'])&~pd.isnull(df_lemma['tense1'])].groupby(['tense1','tense2']):
                    df = group[['form','tense1','tense2','tense_term','CEFR_tense','tense_span','sentence_id']]
                    temp_dict = df.groupby(['form'],as_index=False).size().sort_values(['size'],ascending=[False]).to_dict(orient='list')
                    form_id = {x:i for i,x in enumerate(temp_dict['form'])}
                    temp_dict['tense_term'] = [None]*len(form_id)
                    temp_dict['level'] = [None]*len(form_id)
                    temp_dict['span'] = [None]*len(form_id)
                    temp_dict['sentence_id'] = [None]*len(form_id)
                    for form, group in df.groupby('form',as_index=False):
                        temp_dict['tense_term'][form_id[form]] = group.iloc[0]['tense_term']
                        temp_dict['level'][form_id[form]] = group['CEFR_tense'].values[0]
                        temp_dict['span'][form_id[form]] = group['tense_span'].tolist()
                        temp_dict['sentence_id'][form_id[form]] = group['sentence_id'].astype(int).tolist()
                    tense_count['_'.join(tense)] = temp_dict

            if self.__settings['return_tense_term_count'] or self.__settings['return_tense_stats']:
                tense_term_count = {}

                df_lemma_temp = df_lemma[~pd.isnull(df_lemma['form'])&~pd.isnull(df_lemma['tense_term'])&~pd.isnull(df_lemma['tense1'])&~pd.isnull(df_lemma['tense2'])].copy().reset_index(drop=True)
                
                '''
                infinitive_temp = {}
                infinitive_ids = {}
                for i, row in df_lemma_temp.iterrows():
                    name = row['tense_term']
                    if name.startswith('Infinitive') and 'with modal verb' in name:
                        name_with_level = str(row['CEFR_tense'])+'_'+' '.join(name.split(' ')[:-1])
                        if name_with_level in infinitive_temp:
                            infinitive_temp[name_with_level].append(name.split(' ')[-1])
                            infinitive_ids[name_with_level].append(i)
                        else:
                            infinitive_temp[name_with_level] = [name.split(' ')[-1]]
                            infinitive_ids[name_with_level] = [i]
                for name_with_level,modal_verbs in infinitive_temp.items():
                    name = name_with_level.split('_')[-1] + ' {}'.format(', '.join(set(modal_verbs)))
                    for i in infinitive_ids[name_with_level]:
                        df_lemma_temp.at[i,'tense_term'] = name
                '''
                
                for term,group in df_lemma_temp.groupby('tense_term'):
                    df = group[['form','tense1','tense2','tense_term','CEFR_tense','tense_span','sentence_id']]
                    temp_dict = df.groupby(['form'],as_index=False).size().sort_values(['size'],ascending=[False]).to_dict(orient='list')
                    form_id = {x:i for i,x in enumerate(temp_dict['form'])}
                    temp_dict['tense'] = [None]*len(form_id)
                    temp_dict['level'] = [None]*len(form_id)
                    temp_dict['span'] = [None]*len(form_id)
                    temp_dict['sentence_id'] = [None]*len(form_id)
                    for form, group in df.groupby('form',as_index=False):
                        temp_dict['tense'][form_id[form]] = group['tense1'].values[0]+'_'+group['tense2'].values[0]
                        temp_dict['level'][form_id[form]] = group['CEFR_tense'].values[0]
                        temp_dict['span'][form_id[form]] = group['tense_span'].tolist()
                        temp_dict['sentence_id'][form_id[form]] = group['sentence_id'].astype(int).tolist()
                    tense_term_count[term] = temp_dict

            sum_tense, cumsum_tense = self.count_tense(df_lemma)
            tense_stats = {'sum_token':{'values':list(sum_tense.astype(int))},'cumsum_token':{'values':list(np.round(cumsum_tense.values,4))}}
            a,b,c,d = self.fit_sigmoid(cumsum_tense.values[:-1],np.arange(0,5,0.5))
            tense_stats['cumsum_token']['constants']=[a,b,c,d]
            tense_stats['level'] = {'fit_curve':[self.percentile2level(0.95,a,b,c,d)],
                                    'ninety_five':[self.ninety_five(cumsum_tense,5)],
                                    'fit_error':[self.fit_error(cumsum_tense.values[1:-1],np.arange(0.5,5,0.5),a,b,c,d)]}

            if self.__settings['return_clause_count']:
                clause_count = {}
                for clause,group in df_lemma[~pd.isnull(df_lemma['clause_form'])&~pd.isnull(df_lemma['clause'])].groupby(['clause']):
                    group['span_string'] = group['clause_span'].astype(str)
                    df = group.drop_duplicates(['span_string','sentence_id'])[['clause_form','clause','CEFR_clause','clause_span','sentence_id']]
                    temp_df = df.groupby(['clause_form','CEFR_clause'],as_index=True).agg(len)['sentence_id']#.size().sort_values(['size'],ascending=[False]).to_dict(orient='list')
                    temp_dict = {'clause_form':[x[0]+'_'+str(x[1]) for x in temp_df.index],'size':temp_df.tolist()}
                    form_id = {x:i for i,x in enumerate(temp_dict['clause_form'])}
                    temp_dict['clause_span'] = [None]*len(form_id)
                    temp_dict['sentence_id'] = [None]*len(form_id)
                    for form, group in df.groupby(['clause_form','CEFR_clause'],as_index=False):
                        temp_dict['clause_span'][form_id[form[0]+'_'+str(form[1])]] = group['clause_span'].tolist()
                        temp_dict['sentence_id'][form_id[form[0]+'_'+str(form[1])]] = group['sentence_id'].astype(int).tolist()
                    clause_count[clause] = temp_dict
            
            if self.__settings['return_phrase_count']:
                phrase_count = {}
                if len(dfs_phrase_count)>0:
                    df_phrase_count = pd.concat(dfs_phrase_count)
                    for phrase,group in df_phrase_count.groupby('phrase',as_index=True):
                        group['span_string'] = group['phrase_span'].astype(str)
                        group = group.drop_duplicates(['span_string','sentence_id'])
                        temp_df = group.agg(len)['sentence_id']
                        temp_dict = {'id':phrase_original2id.get(phrase,0),'phrase_ambiguous':group['phrase_ambiguous'].tolist()[0],'size':temp_df.tolist(),'phrase_span':group['phrase_span'].tolist(),'phrase_confidence':group['phrase_confidence'].tolist(),'sentence_id':group['sentence_id'].astype(int).tolist()}
                        phrase_count[phrase] = temp_dict

            mean_clause = n_clausal and n_clauses/n_clausal or 0

            if len(clause_levels)>0:
                clause_level = round(min(np.percentile(clause_levels,90),6),1)
            else:
                clause_level = 0

            mean_length = len(dfs) and n_words/len(dfs) or 0

            clause_stats = {'p_clausal':len(dfs) and n_clausal/len(dfs) or 0,'mean_clause':mean_clause,'mean_length':mean_length,'level':clause_level,'n_words':n_words}
            
            sum_series_token, cumsum_series_token, sum_series_type, cumsum_series_type = self.count_cefr(df_lemma)
            
            stats_dict = {'sum_token':{'values':list(sum_series_token.astype(int))},
                        'cumsum_token':{'values':list(np.round(cumsum_series_token.values,4))},
                        'sum_type':{'values':list(sum_series_type.astype(int))},
                        'cumsum_type':{'values':list(np.round(cumsum_series_type.values,4))}}
            
            for k in set(['cumsum_token','cumsum_type']):
                a,b,c,d = self.fit_sigmoid((stats_dict[k]['values'][1:-1]),range(6))
                stats_dict[k]['constants']=[a,b,c,d]
                if k == 'cumsum_token':
                    stats_dict['level'] = {'fit_curve':[self.percentile2level(0.95,a,b,c,d)],
                                        'ninety_five':[self.ninety_five(cumsum_series_token)],
                                        'fit_error':[self.fit_error(stats_dict[k]['values'][2:-1],range(1,6),a,b,c,d)]}
            
            if self.__settings['return_final_levels'] or self.__settings['return_tense_stats']:
                if tense_stats["level"]["fit_error"][0]>=0.05:
                    tense_level = tense_stats["level"]["ninety_five"][0]
                else:
                    tense_level = tense_stats["level"]["fit_curve"][0]

                if stats_dict["level"]["fit_error"][0]>=0.1:
                    vocabulary_level =stats_dict["level"]["ninety_five"][0]
                else:
                    vocabulary_level = stats_dict["level"]["fit_curve"][0]
                
                
                
                average_level = (vocabulary_level+tense_level+clause_level)/3
                general_level = max([vocabulary_level,tense_level,average_level,clause_level-0.5])

                final_levels = {'general_level':round(general_level,1),'vocabulary_level':round(vocabulary_level,1),'tense_level':round(tense_level,1),'clause_level':clause_level}

            result_dict = {}
            if self.__settings['return_sentences']:
                result_dict['sentences'] = sentences
                #self.sentences = sentences
            if self.__settings['return_wordlists']:
                result_dict['wordlists'] = wordlists
                #self.wordlists = wordlists
            if self.__settings['return_vocabulary_stats']:
                result_dict['stats'] = stats_dict
                #self.vocabulary_stats = stats_dict
            if self.__settings['return_tense_count']:
                result_dict['tense_count'] = tense_count
                #self.tense_count = tense_count
            if self.__settings['return_tense_term_count']:
                result_dict['tense_term_count'] = tense_term_count
            if self.__settings['return_tense_stats']:
                if len(tense_count)>0:
                    df_tense_stats = pd.DataFrame([{'tense':tense,'level':d['level'][0],'count':sum(d['size'])} for tense, d in tense_count.items()])
                    df_tense_stats['ratio'] = np.round(df_tense_stats['count']/sum(df_tense_stats['count']),2)
                    tense_stats['tense_summary'] = df_tense_stats.sort_values('count',ascending=False).to_dict('list')
                else:
                    tense_stats['tense_summary'] = {}
                if len(tense_term_count)>0:
                    df_tense_term_stats = pd.DataFrame([{'tense_term':term,'level':d['level'][0],'count':sum(d['size'])} for term, d in tense_term_count.items()])
                    df_tense_term_stats['level_diff'] = np.round(general_level-df_tense_term_stats['level'],1)
                    df_tense_term_stats['ratio'] = np.round(df_tense_term_stats['count']/sum(df_tense_term_stats['count']),2)
                    tense_stats['tense_term_summary'] = df_tense_term_stats.sort_values(['level_diff','count'],ascending=[True,False])[['tense_term','level','level_diff','count','ratio']].to_dict('list')
                else:
                    tense_stats['tense_term_summary'] = {}
                result_dict['tense_stats'] = tense_stats
                #self.tense_stats = tense_stats
            if self.__settings['return_clause_count']:
                result_dict['clause_count'] = clause_count
                #self.clause_count = clause_count
            if self.__settings['return_clause_stats']:
                result_dict['clause_stats'] = clause_stats
                #self.clause_stats = clause_stats
            if self.__settings['return_phrase_count']:
                result_dict['phrase_count'] = phrase_count
                #self.clause_count = clause_count
            if self.__settings['return_final_levels']:
                result_dict['final_levels'] = final_levels
                #self.final_levels = final_levels
            self.result = result_dict

        def sum_cumsum(self,df_lemma,mode='token'):
            base = pd.Series(dict(zip(range(-1,7),[0.]*8)))
            if mode == 'type':
                df_lemma = df_lemma.drop_duplicates(['lemma','pos'])
            counts = base.add(df_lemma.groupby('CEFR')['word'].agg(len),fill_value=0.)
            if counts.sum()==0:
                p = base
            else:
                p = counts/counts.sum()
            return counts, p.cumsum()

        def count_tense(self,df_lemma):
            base = pd.Series(dict(zip(np.arange(0,5.5,0.5),[0.]*11)))
            counts = base.add(df_lemma[~pd.isnull(df_lemma['CEFR_tense'])].groupby('CEFR_tense')['word'].agg(len),fill_value=0.)
            if counts.sum()==0:
                p = base
            else:
                p = counts/counts.sum()
            return counts, p.cumsum()

        def fit_sigmoid(self,cumsum_series_values,levels):
            # [1:-1]
            x = cumsum_series_values
            y = levels

            def func(x, a, b, c, d):
                return a/(-b+np.exp(-x*c+d))

            try:
                popt, _ = curve_fit(func, x, y, maxfev=10000,bounds=(0.00001,np.inf))
                a,b,c,d = popt
            except:
                a = 0.00001
                b = 0.00001
                c = 0.00001
                d = 0.00001
            return a,b,c,d

        def fit_error(self,cumsum_series_values,levels,a,b,c,d):
            pred = np.maximum(0,self.level2percentile(levels,a,b,c,d))
            return np.sqrt(np.nanmean((cumsum_series_values-pred)**2))

        def percentile2level(self,x,a,b,c,d):
            return a/(-b+np.exp(-x*c+d))

        def level2percentile(self,y,a,b,c,d):
            y = np.array(y)
            return (d-np.log(a/y+b))/c

        def ninety_five(self,cumsum_series,default=6):
            if cumsum_series.sum()==0:
                return 0
            level = default
            for i,v in cumsum_series.iteritems():
                if v>=0.95:
                    level = i
                    break
            return level

        def sentence_clause_level(self,levels):
            if len(levels) == 0:
                return 0
            levels = reversed(sorted(levels))
            X = 0
            for i, x in enumerate(levels):
                X += np.exp(-i)*x
            return X

        def count_cefr(self,df_lemma):
            df_lemma = df_lemma[(df_lemma['pos']!='PUNCT')&(df_lemma['pos']!='SPACE')]
            sum_series_token, cumsum_series_token = self.sum_cumsum(df_lemma)
            sum_series_type, cumsum_series_type = self.sum_cumsum(df_lemma,mode='type')
            return sum_series_token, cumsum_series_token, sum_series_type, cumsum_series_type

        def start_analyze(self, propn_as_lowest=True,intj_as_lowest=True,keep_min=True,
                        return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                        return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                        return_clause_stats=True,return_phrase_count=True,return_final_levels=True):
            self.__settings['propn_as_lowest']=propn_as_lowest
            self.__settings['intj_as_lowest']=intj_as_lowest
            self.__settings['keep_min']=keep_min
            self.__settings['return_sentences']=return_sentences
            self.__settings['return_wordlists']=return_wordlists
            self.__settings['return_vocabulary_stats']=return_vocabulary_stats
            self.__settings['return_tense_count']=return_tense_count
            self.__settings['return_tense_term_count']=return_tense_term_count
            self.__settings['return_tense_stats']=return_tense_stats
            self.__settings['return_clause_count']=return_clause_count
            self.__settings['return_clause_stats']=return_clause_stats
            self.__settings['return_final_levels']=return_final_levels
            self.__settings['return_phrase_count']=return_phrase_count
            self.process()

        def print_settings(self):
            return self.__settings

    class ReadabilityAnalyzer(object):
        def __init__(self,outer):
            self.shared_object = outer
            self.result = None

        def start_analyze(self,language='en'):
            '''
            To be converted to grade:
            textstat.flesch_reading_ease(text)
            textstat.dale_chall_readability_score(text)
            textstat.spache_readability(text)
            
            Grade:
            textstat.flesch_kincaid_grade(text)
            textstat.gunning_fog(text)
            textstat.smog_index(text)
            textstat.automated_readability_index(text)
            textstat.coleman_liau_index(text)
            textstat.linsear_write_formula(text)
            textstat.text_standard(text, float_output=False)
            
            Foreigner learner:
            textstat.mcalpine_eflaw(text)

            Others:
            textstat.reading_time(text, ms_per_char=14.69)
            textstat.lexicon_count(text, removepunct=True)
            textstat.sentence_count(text)

            Spanish:
            textstat.fernandez_huerta(text)
            textstat.szigriszt_pazos(text)
            textstat.gutierrez_polini(text)
            textstat.crawford(text)

            Italian:
            textstat.gulpease_index(text)
            '''
            textstat.set_lang(language)
            textstat.set_rounding(False)

            text = self.shared_object.text
            if language=='es':
                self.result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'fernandez_huerta':textstat.fernandez_huerta(text),
                        'szigriszt_pazos':textstat.szigriszt_pazos(text),
                        'gutierrez_polini':textstat.gutierrez_polini(text),
                        'crawford':textstat.crawford(text),
                        'lexicon_count':textstat.lexicon_count(text, removepunct=True),
                        'sentence_count':textstat.sentence_count(text)}
            elif language=="it":
                self.result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'gulpease_index':textstat.gulpease_index(text),
                        'lexicon_count':textstat.lexicon_count(text, removepunct=True),
                        'sentence_count':textstat.sentence_count(text)}
            elif language=="pl":
                self.result = {'gunning_fog':textstat.gunning_fog(text),
                        'lexicon_count':textstat.lexicon_count(text, removepunct=True),
                        'sentence_count':textstat.sentence_count(text)}
            elif language in set(['de','fr','nl','ru']):
                self.result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'lexicon_count':textstat.lexicon_count(text, removepunct=True),
                        'sentence_count':textstat.sentence_count(text)}
            else:
                result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'flesch_kincaid_grade':textstat.flesch_kincaid_grade(text),
                        'gunning_fog':textstat.gunning_fog(text),
                        'smog_index':textstat.smog_index(text),
                        'dale_chall_readability_score':textstat.dale_chall_readability_score(text),
                        'automated_readability_index':textstat.automated_readability_index(text),
                        'coleman_liau_index':textstat.coleman_liau_index(text),
                        'linsear_write_formula':textstat.linsear_write_formula(text),
                        #'text_standard':textstat.text_standard(text, float_output=True),
                        'spache_readability':textstat.spache_readability(text),
                        'mcalpine_eflaw':textstat.mcalpine_eflaw(text),
                        #'reading_time':textstat.reading_time(text, ms_per_char=14.69),
                        'lexicon_count':textstat.lexicon_count(text, removepunct=True),
                        'sentence_count':textstat.sentence_count(text)}
                result['readability_consensus'] = (result["flesch_kincaid_grade"]+result["gunning_fog"]+result["smog_index"]+result["automated_readability_index"]+result["coleman_liau_index"]+result["linsear_write_formula"]+(result["dale_chall_readability_score"]*2-5))/7
                self.result = result

    class CefrSimplifier(object):

        def __init__(self, outer):
            self.shared_object = outer
            self.result = None

        def start_simplify(self, text, target_level, target_adjustment=0.5, n=1, by_sentence=False, auto_retry=False, up=False):
            n_tokens = len(gpt_tokenizer.encode(text))

            n_pieces = 1
            while n_tokens/n_pieces>1500:
                n_pieces += 1
            mean_piece_length = n_tokens/n_pieces
                
            if up:
                n = min(n, int((4000-mean_piece_length)/(mean_piece_length*0.75)))
            else:
                n = min(n, int((4000-mean_piece_length)/(mean_piece_length*1.25)))

            pieces = []
            if n_pieces>1:
                result = self.shared_object.analyze_cefr(text,return_sentences=True, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                
                piece = ''
                for _,v in result['sentences'].items():
                    for i in range(len(v['whitespace'])):
                        piece += v['word'][i]+' '*v['whitespace'][i]
                    if len(gpt_tokenizer.encode(piece))>=mean_piece_length:
                        pieces.append(piece)
                        piece = ''
                if piece!='':
                    pieces.append(piece)
                
            else:
                result = self.shared_object.analyze_cefr(text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                pieces.append(text)

            before_levels = result['final_levels']
            
            simplifications = []
            for piece in pieces:
                try:
                    candidates = self.get_simplification(piece, target_level=target_level, target_adjustment=target_adjustment, n=n, by_sentence=by_sentence, up=up)
                except Exception as e:
                    return {'error':e.__class__.__name__,'detail':str(e)}
                min_difference = 100
                for candidate in candidates:
                    result = self.shared_object.analyze_cefr(candidate,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                    return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                    return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                    difference = result['final_levels']['general_level']-(target_level+target_adjustment)
                    if difference==0:
                        simplification = candidate
                        after_levels = result['final_levels']
                        break
                    elif abs(difference)<min_difference or abs(difference)==min_difference and difference>0:
                        simplification = candidate
                        after_levels = result['final_levels']
                        min_difference = abs(difference)
                simplifications.append(simplification)
                
            after_text = ' '.join(simplifications)
            
            if n_pieces>1:
                after_levels = self.shared_object.analyze_cefr(text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)['final_levels']
                
            if auto_retry and int(after_levels['general_level'])!=target_level:
                return self.start_simplify(text, target_level, target_adjustment=target_adjustment, n=n, by_sentence=by_sentence, auto_retry=False, up=up)

            self.result = {'simplified':after_text, 'before':before_levels, 'after': after_levels}

        def get_simplification(self, text, target_level, target_adjustment=0.5, n=1, by_sentence=False, up=False):
            if up:
                int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
                max_length = int(round(np.log(target_level+1+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))

                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", n=n,
                messages=[{"role": "user", "content": f"Rewrite this passage to make it more complex. Use mainly words at CEFR {int2cefr[target_level]} levels. Write sentences with {min_length} to {max_length} words.\nPassage: " + text}]
                )
            else:
                int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
                max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment-1+1.5)/np.log(1.1),0))

                if target_level>0:
                    levels = [int2cefr[i] for i in range(target_level+1)]
                    levels = ', '.join(levels[:-1]) + f' and {levels[-1]}'
                    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", n=n,
                    messages=[{"role": "user", "content": f"Rewrite this passage to improve its readability. Use mainly words at CEFR {levels} levels. Write sentences with {min_length} to {max_length} words. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences. \nPassage: " + text}]
                    )
                else:
                    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", n=n,
                    messages=[{"role": "user", "content": f"Rewrite this passage to improve its readability. Use only words at CEFR A1 level. Write sentences with less than {max_length} words. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.\nPassage: " + text}]
                    )

            simplifications = []
            for x in completion['choices']:
                x = x['message']['content'].strip()
                if x.lower().startswith("simplified: "):
                    x = x[12:].strip()
                elif x.lower().startswith("simplified version: "):
                    x = x[20:].strip()
                simplifications.append(x)
            return simplifications

    class CefrAdaptor(object):

        def __init__(self, outer):
            self.shared_object = outer
            self.result = None

        def start_adapt(self, text, target_level, target_adjustment=0.5, even=False, n=1, auto_retry=False):

            n_tokens = len(gpt_tokenizer.encode(text))

            n_pieces = 1
            while n_tokens/n_pieces>1500:
                n_pieces += 1
            mean_piece_length = n_tokens/n_pieces
                
            pieces = []
            if n_pieces>1:
                result = self.shared_object.analyze_cefr(text,return_sentences=True, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)

                piece = ''
                for _,v in result['sentences'].items():
                    for i in range(len(v['whitespace'])):
                        piece += v['word'][i]+' '*v['whitespace'][i]
                    if len(gpt_tokenizer.encode(piece))>=mean_piece_length:
                        pieces.append(piece)
                        piece = ''
                if piece!='':
                    pieces.append(piece)
                
            else:
                result = self.shared_object.analyze_cefr(text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                pieces.append(text)

            before_levels = result['final_levels']
            
            adaptations = []
            for piece in pieces:
                if int(before_levels['vocabulary_level'])>target_level:
                    change_vocabulary = -1
                elif int(before_levels['vocabulary_level'])<target_level and even:
                    change_vocabulary = 1
                else:
                    change_vocabulary = 0

                if int(before_levels['clause_level'])>target_level:
                    change_clause = -1
                    new_n = min(n, int((4000-mean_piece_length)/(mean_piece_length*0.75)))
                elif int(before_levels['clause_level'])<target_level and even:
                    change_clause = 1
                    new_n = min(n, int((4000-mean_piece_length)/(mean_piece_length*1.25)))
                else:
                    change_clause = 0
                    new_n = n

                try:
                    if change_vocabulary==0 and change_clause==0:
                        adaptations.append(piece)
                        continue
                    else:
                        candidates = self.get_adaptation(
                            piece, target_level=target_level, target_adjustment=target_adjustment, n=new_n, 
                            change_vocabulary=change_vocabulary, change_clause=change_clause)
                except Exception as e:
                    self.result = {'error':e.__class__.__name__,'detail':str(e)}
                    return

                min_difference = 100
                min_difference_std = -1
                for candidate in candidates:
                    result = self.shared_object.analyze_cefr(candidate,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                    return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                    return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)

                    if change_vocabulary:
                        vocabulary_difference = abs(result['final_levels']['vocabulary_level']-(target_level+target_adjustment))
                    else:
                        vocabulary_difference = abs(result['final_levels']['vocabulary_level']-before_levels['vocabulary_level'])

                    tense_difference = abs(result['final_levels']['tense_level']-(target_level+target_adjustment))

                    if change_clause:
                        clause_difference = abs(result['final_levels']['clause_level']-(target_level+target_adjustment))
                    else:
                        clause_difference = abs(result['final_levels']['clause_level']-before_levels['clause_level'])

                    difference = vocabulary_difference+tense_difference*0.5+clause_difference
                    difference_std = np.std([vocabulary_difference,tense_difference*0.5,clause_difference])
                    if difference<1:
                        adaptation = candidate
                        after_levels = result['final_levels']
                        break
                    elif difference<min_difference or difference==min_difference and difference_std<min_difference_std:
                        adaptation = candidate
                        after_levels = result['final_levels']
                        min_difference = difference
                        min_difference_std = difference_std
                adaptations.append(adaptation)
                
            after_text = ' '.join(adaptations)
            
            if n_pieces>1:
                after_levels = self.shared_object.analyze_cefr(text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)['final_levels']
                
            if auto_retry and int(after_levels['general_level'])!=target_level:
                return self.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, auto_retry=False)

            self.result = {'adaptation':after_text, 'before':before_levels, 'after': after_levels}

        def get_adaptation(self, text, target_level, target_adjustment=0.5, n=1, change_vocabulary=-1, change_clause=-1):
            int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
            levels = [int2cefr[i] for i in range(target_level+1)]
            levels = ', '.join(levels[:-1]) + f' and {levels[-1]}'
            if change_clause<0:
                max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment-1+1.5)/np.log(1.1),0))
                if change_vocabulary<0:
                    prompt = f"Rewrite this passage to improve its readability. Use mainly words at CEFR {levels} levels. Write sentences with {min_length} to {max_length} words. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences."
                elif change_vocabulary>0:
                    prompt = f"Rewrite this passage to replace easy words so that most of the passage uses words at CEFR {int2cefr[target_level]} level. Write sentences with {min_length} to {max_length} words. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences."
                else:
                    prompt = f"Rewrite this passage to improve its readability. Write sentences with {min_length} to {max_length} words. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences. Use only the vocabulary in the original passage."
            elif change_clause>0:
                max_length = int(round(np.log(target_level+1+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
                if change_vocabulary<0:
                    prompt = f"Rewrite this passage to make it more complex. Write sentences with {min_length} to {max_length} words. Uses only words at CEFR {levels} levels."
                elif change_vocabulary>0:
                    prompt = f"Rewrite this passage to make it more complex. Use mainly words at CEFR {int2cefr[target_level]} level. Write sentences with {min_length} to {max_length} words."
                else:
                    prompt = f"Rewrite this passage to make it more complex. Write sentences with {min_length} to {max_length} words. Use only the vocabulary in the original passage."
            else:
                if change_vocabulary<0:
                    prompt = f"In this passage, replace difficult words so that most of the passage uses only words at CEFR {levels} levels."
                elif change_vocabulary>0:
                    prompt = f"In this passage, replace easy words so that most of the passage uses words at CEFR {int2cefr[target_level]} level."
                else:
                    return []

            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", n=n,
            messages=[{"role": "user", "content": prompt+"\nPassage: " + text}]
            )

            adaptations = []
            for x in completion['choices']:
                x = x['message']['content'].strip()
                if x.lower().startswith("simplified: "):
                    x = x[12:].strip()
                elif x.lower().startswith("simplified version: "):
                    x = x[20:].strip()
                adaptations.append(x)
            return adaptations