import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from . import spacy
from . import word as solar_word
from . import modify_text
from .edit_distance_modified import edit_distance
import pickle, re, tensorflow, textstat, warnings, time, youtube_dl, requests, torch, httpx
from textacy import text_stats
from collections import Counter
import Levenshtein as lev
from lexical_diversity import lex_div as ld
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import curve_fit, fsolve
from scipy.stats import percentileofscore
from .lemmatizers import fine_lemmatize
from .utils import InformError
from nltk.stem import WordNetLemmatizer, LancasterStemmer, PorterStemmer
from nltk.corpus import stopwords
from openai import OpenAI
from ftlangdetect import detect
from sentence_transformers import SentenceTransformer
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

pd.set_option('future.no_silent_downcasting', True)

# Lexile files
neural_model = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/lexile_20220410_2.h5'),compile=False)
neural_model2 = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/lexile_20240208'),compile=False)
sentence_level_model = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/cefr/sentence_level_model'),compile=False)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

with open(os.path.join(BASE_DIR, 'files/model_files/lexile_linear.pkl'), 'rb') as file:
    linear_model = pickle.load(file)

with open(os.path.join(BASE_DIR, 'files/model_files/lexile_linear_20240208.pkl'), 'rb') as file:
    linear_model2 = pickle.load(file)

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

df_cefr_prim = pd.DataFrame(cefr_w_pos_min_prim, index=['level']).T.rename_axis(['lemma','pos']).reset_index()
df_cefr = pd.DataFrame(cefr_w_pos, index=['level']).T.rename_axis(['lemma','pos']).reset_index()
df_cefr_sup = pd.DataFrame(cefr_w_pos_sup, index=['level']).T.rename_axis(['lemma','pos']).reset_index()    
wordlist_dicts = [df_cefr_prim,df_cefr,df_cefr_sup]


df_reference_words = pd.concat([pd.DataFrame(pd.Series(cefr_w_pos_min_prim)).reset_index(),pd.DataFrame(pd.Series(cefr_w_pos)).reset_index()]).drop_duplicates(['level_0','level_1'])
df_temp = df_reference_words[df_reference_words['level_0'].apply(lambda x: str(x).lower().endswith('e') and len(str(x))>=4)].copy()
df_temp['level_0'] = [x[:-1] for x in df_temp['level_0'].values]
df_reference_words = pd.concat([df_reference_words,df_temp]).drop_duplicates(['level_0','level_1']).reset_index(drop=True)

df_reference_words_sup = pd.DataFrame(pd.Series(cefr_w_pos_sup)).reset_index()
df_temp = df_reference_words_sup[df_reference_words_sup['level_0'].apply(lambda x: str(x).lower().endswith('e') and len(str(x))>=4)].copy()
df_temp['level_0'] = [x[:-1] for x in df_temp['level_0'].values]
df_reference_words_sup = pd.concat([df_reference_words_sup,df_temp]).drop_duplicates(['level_0','level_1']).reset_index(drop=True)

cefr_word_model = tensorflow.keras.models.load_model(os.path.join(BASE_DIR, 'files/model_files/cefr_word_model.h5'))

ignore_phrases = set(["risk of (some inclement weather)"])
df_phrases = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/phrases.pkl'),'rb'))
df_phrases = df_phrases[~((df_phrases['characters']<=9)&(df_phrases['length']<=2)&(df_phrases['level']<=1)&df_phrases['word'].apply(lambda x: x in set(['have','and','do','it','or','on','so','at','you','after','in','down','i','up','that','to'])))][['id','original','original_to_display','clean','followed_by','lemma','pos','dep','word','is_idiom','ambiguous','phrase_parts']]
df_phrases = df_phrases[df_phrases['original'].apply(lambda x: x not in ignore_phrases)]
phrase_original2id = df_phrases.set_index('original')['id'].to_dict()
people_list = set(pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/people_list.pkl'),'rb')))
topic_words = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/cefr/topic_words.pkl'),'rb'))

del br2am, df_corpus, df_aoa, df_mrc, df_abstract, df_temp

aoa_dict = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/aoa_dict.pkl'),'rb'))
freq_dict_w_pos = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/log_frequency_dict_w_pos.pkl'),'rb'))
freq_dict_wo_pos = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/log_frequency_dict_wo_pos.pkl'),'rb'))
concreteness_dict = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/concreteness_dict.pkl'),'rb'))


nlp = spacy.load('en_core_web_trf')

stemmers = [PorterStemmer(), LancasterStemmer()]
lemmatizer = WordNetLemmatizer()


class AdoTextAnalyzer(object):
    def __init__(self, openai_api_key=None):
        self.init()
        self.openai_api_key = openai_api_key
        self.detect = detect
        self.v = 1

    def init(self):
        self.text = None
        self.doc = None
        self.cefr = self.CEFRAnalyzer(self)
        self.cefr2 = self.CEFRAnalyzer2(self)
        self.readability = self.ReadabilityAnalyzer(self)
        self.catile = self.CatileAnalyzer(self)
        self.catile2 = self.CatileAnalyzer2(self)
        self.simplifier = None
        self.adaptor = None

    def make_doc(self):
        self.doc = nlp(self.text)
        for x in self.doc:
            x = fine_lemmatize(x,self.doc,nlp)

    def analyze_cefr(self,text,
                     settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}},
                     outputs=['final_levels'],
                     propn_as_lowest=True,intj_as_lowest=True,keep_min=True,custom_dictionary={},
                     return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                     return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                     return_clause_stats=True,return_phrase_count=True,return_final_levels=True,
                     return_result=True,clear_simplifier=True,return_modified_final_levels=False, v=1):
        text = self.clean_text(text)

        if v==1:

            # if detect(text.replace('\n',' '))['lang'] != 'en':
            #     raise InformError("Language not supported. Please use English.")

            temp_settings = {'propn_as_lowest':propn_as_lowest,'intj_as_lowest':intj_as_lowest,'keep_min':keep_min,
                            'return_sentences':return_sentences, 'return_wordlists':return_wordlists,'return_vocabulary_stats':return_vocabulary_stats,
                            'return_tense_count':return_tense_count,'return_tense_term_count':return_tense_term_count,'return_tense_stats':return_tense_stats,'return_clause_count':return_clause_count,
                            'return_clause_stats':return_clause_stats,'return_phrase_count':return_phrase_count,'return_final_levels':return_final_levels,'return_modified_final_levels':return_modified_final_levels}

            if self.v!=v or text!=self.text or custom_dictionary!={} or temp_settings!=self.cefr.print_settings():
                self.init()
                self.v = v
                self.text = text

            if self.doc is None:
                self.make_doc()
            self.cefr = self.CEFRAnalyzer(self)

            clean_custom_dictionary = {}
            for k, l in default_settings['custom_dictionary'].items():
                if isinstance(l, str):
                    l = cefr2float(l)
                if l is None:
                    continue
                if isinstance(k, tuple) or isinstance(k, list):
                    key = tuple([k[0].lower(),standardisePos(k[1])])
                    clean_custom_dictionary[key] = l
                else:
                    clean_custom_dictionary[k] = l
            self.cefr.start_analyze(propn_as_lowest=propn_as_lowest,intj_as_lowest=intj_as_lowest,keep_min=keep_min,custom_dictionary=clean_custom_dictionary,
                        return_sentences=return_sentences, return_wordlists=return_wordlists,return_vocabulary_stats=return_vocabulary_stats,
                        return_tense_count=return_tense_count,return_tense_term_count=return_tense_term_count,return_tense_stats=return_tense_stats,return_clause_count=return_clause_count,
                        return_clause_stats=return_clause_stats,return_phrase_count=return_phrase_count,return_final_levels=return_final_levels,
                        return_modified_final_levels=return_modified_final_levels)

            if return_result:
                return self.cefr.result
            
        else:
            all_outputs = {'all','sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','phrase_count','final_levels','modified_final_levels'}
            invalid_outputs = set(outputs)-all_outputs
            if len(invalid_outputs)>0:
                raise InformError(f"Invalid output type: {', '.join(invalid_outputs)}")
            all_settings = {'propn_as_lowest','intj_as_lowest','keep_min','custom_dictionary','as_wordlist'}
            invalid_settings = set(settings.keys())-all_settings
            if len(invalid_settings)>0:
                raise InformError(f"Invalid setting: {', '.join(invalid_settings)}")
            
            if self.v!=v or text!=self.text or self.cefr2.outputs != outputs or self.cefr2.settings != settings:
                self.init()
                self.v = v
                self.text = text
                
                if 'all' in outputs:
                    self.cefr2.outputs = {'sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','phrase_count','final_levels','modified_final_levels'}
                else:
                    self.cefr2.outputs = set(outputs)
                default_settings = {'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}}
                default_settings.update(settings)

                clean_custom_dictionary = {}
                for k, l in default_settings['custom_dictionary'].items():
                    if isinstance(l, str):
                        l = cefr2float(l)
                    if l is None:
                        continue
                    if isinstance(k, tuple) or isinstance(k, list):
                        key = tuple([k[0].lower(),standardisePos(k[1])])
                        clean_custom_dictionary[key] = l
                    else:
                        clean_custom_dictionary[k] = l
                default_settings['custom_dictionary'] = clean_custom_dictionary
                self.cefr2.settings = default_settings
                self.make_doc()
            if self.doc is None:
                self.make_doc()
            self.cefr2.start_analyze()
            self.cefr.result = self.cefr2.result
            return self.cefr.result
        

    def analyze_readability(self,text,language='en',return_grades=False,return_result=True):
        text = self.clean_text(text)
        # if language=='en':
        #     detected_language = detect(text.replace('\n',' '))['lang']
        #     if detected_language not in ['es',"it","pl",'de','fr','nl','ru','en']:
        #         raise InformError("Language not supported.")
        #     else:
        #         language = detected_language

        if text!=self.text or self.readability is None or self.readability.return_grades!=return_grades:
            self.init()
            self.text = text
        if self.readability is None:
            self.readability = self.ReadabilityAnalyzer(self)
        self.readability.return_grades = return_grades
        self.readability.start_analyze(language)
        if return_result:
            return self.readability.result

    def analyze_catile(self,text,return_result=True,v=1):

        # if detect(text.replace('\n',' '))['lang'] != 'en':
        #     raise InformError("Language not supported. Please use English.")
            
        text = self.clean_text(text)
        if self.v!=v or text!=self.text:
            self.init()
            self.v = v
            self.text = text
        if self.doc is None:
            self.make_doc()
        if v==1:
            self.catile.start_analyze()
        else:
            self.catile2.start_analyze()
            self.catile.result = self.catile2.result
        if return_result:
            return self.catile.result


    def clean_text(self, text):
        return text.replace("\u00A0", " ").replace('\xa0',' ').strip()

    class CatileAnalyzer2(object):

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

        def high_avg(self, a, p=3):
            if len(a) == 0:
                return 0
            a = np.array(a)
            return np.mean(a**p)**(1/p)

        def get_ability_2(self, data, ab_at_arr, n_bins = 'auto', n_iter = 1):
            # sort the requested abilities
            ab_at_arr = np.sort(ab_at_arr)
            
            # remove all zeros
            data = np.array(data)
            data = data[data != 0]

            if str(n_bins) != 'auto':
                h, data_hist = np.histogram(data, bins = n_bins, density=True)  
            else:
                h, data_hist = np.histogram(data, bins = 'auto', density=True)
        
            h_csum = np.cumsum(h)
            h_csum_norm = h_csum / h_csum.max()

            est_x = np.nanmean(data_hist)

            init_vals = [1/est_x,est_x,0]
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
            arr = np.array(arr)
            arr = arr[arr != 0]
            
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
            return c + ((1-c) / (1+np.exp(np.float128(a*(b-x)))))

        def rasch_func_solve(self, x, arr):
            a = arr[0]
            b = arr[1]
            c = arr[2]
            y = arr[3]
            return c + ((1-c) / (1+np.exp(np.float128(a*(b-x))))) - y

        def process_text(self):

            #text = standardize_2(text)
            doc = self.shared_object.doc
            #run_time('nlp')
            
            # Split text
            sent_groups = []
            group = []
            n_words_temp = 0
            for sent in list(doc.sents):
                if not (bool(re.search('[a-zA-Z]', sent.text)) and len(sent.text.strip())>1):
                    continue
                n_words_temp += text_stats.api.TextStats(sent).n_words
                group.append(sent)
                if n_words_temp>=125:
                    sent_groups.append(group)
                    group = []
                    n_words_temp = 0
                    
            if len(group)>0:
                if len(sent_groups)>0:
                    sent_groups[-1] += group
                else:
                    sent_groups.append(group)
            
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
            dfs_all = []
            dfs_500 = []
            summaries = []
            sent_depth_list = []
            sent_length_list_all = []
            n_sents = 0
            
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
            
            for i_sent_group, sent_group in enumerate(sent_groups):
                for sent in sent_group:
                    n_sents += 1
                    for x in sent:
                        original_list.append(x.orth_)
                        dep_list.append(x.dep_)
                        pos_list.append(x.pos_)
                        morph_list.append(nlp.explain(x.tag_))
                        lemma_list.append(x.lemma_)
                        if lemma_list[-1] in most_freq_50 or x.is_stop:
                            stop_list.append(True)
                        else:
                            stop_list.append(False)

                        # Frequency, AOA, ABSTRACTNESS
                        am_lemma = br2am_dict.get(x.lemma_,x.lemma_)
                        cefr = self.shared_object.cefr2.get_word_cefr(am_lemma, x.orth_.lower())
                        if x.pos_ in {'PROPN','INTJ'}:
                            freq_list.append(freq_dict_w_pos.get((am_lemma,x.pos_),freq_dict_wo_pos.get(am_lemma,1)))
                            aoa_list.append(aoa_dict.get(x.lemma_,0))
                            abstract_list.append(1-concreteness_dict.get(x.lemma_,1))
                        else:
                            freq_list.append(freq_dict_w_pos.get((am_lemma,x.pos_),freq_dict_wo_pos.get(am_lemma,1-cefr/6)))
                            aoa_list.append(aoa_dict.get(x.lemma_,cefr/6))
                            abstract_list.append(1-concreteness_dict.get(x.lemma_,1-cefr/6))

                        if stop_list[-1]:
                            freq_list_no_stop.append(0)
                        else:
                            freq_list_no_stop.append(freq_list[-1])

                        # NUMBER OF SYLLABLES and DECODABILITY
                        syllable_count = solar_word.count_syllables(original_list[-1].lower())
                        nsyl_list.append(syllable_count)
                        if syllable_count > 2:
                            decode_lemma_list.append(9)
                            decode_original_list.append(9)
                        else:
                            decode_lemma_list.append(solar_word.decoding_degree(lemma_list[-1].lower()))
                            decode_original_list.append(solar_word.decoding_degree(original_list[-1].lower()))
                        
                        sent_index_list.append(n_sents-1)

                    sent_depth_list.append(self.walk_tree(sent.root, 0))
                    sent_list.append(x.sent.text.lower().strip())
                    sent_length_list.append(solar_word.count_syllables(sent.text))

                    length = text_stats.api.TextStats(sent).n_words
                    if length>longest_length:
                        longest_sent = sent.text.strip()
                        longest_length = length
                    n_words += length


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
                
                noncompressibility = self.get_noncompressibility(' '.join(df['original'].values).replace('\n',' ').replace('  ',' ').strip())
                        
                df = df[(df['pos']!='SPACE')&(df['pos']!='PUNCT')]

                tokens = df['lemma'].values
                hdd = ld.hdd(tokens)
                mtld = ld.mtld_ma_wrap(tokens)

                if len(tokens):
                    maas_ttr = ld.maas_ttr(tokens)
                else:
                    maas_ttr = 0


                df_freq_type = df.drop_duplicates(['lemma','pos'])
                mean_log_freq_type = np.mean(df_freq_type['frequency'])
                df_freq_type_no_stop = df[~df['is_stop']].drop_duplicates(['lemma','pos'])
                mean_log_freq_type_no_stop = np.mean(df_freq_type_no_stop['frequency'])
                mean_log_freq_token = np.mean(df['frequency'])
                df_freq_token_no_stop = df[~df['is_stop']]
                mean_log_freq_token_no_stop = np.mean(df_freq_token_no_stop['frequency'])
                        
                # Longest common string
                for i in range(len(sent_list)):
                    couplet = sent_list[i:min(i+2,len(sent_list))]
                    triplet = sent_list[i:min(i+3,len(sent_list))]
                    if len(couplet) == 2:
                        lcs2_list.append(self.lcs(couplet))
                    if len(triplet) == 3:
                        lcs3_list.append(self.lcs(triplet))

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
                if n_words>=500:
                    dfs_500.append(pd.concat(dfs))
                    dfs = []
                dfs_all.append(df)
                summaries.append(summary)

                sent_length_list_all += sent_length_list

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
                sent_index_list = []
                freq_list = []
                freq_list_no_stop = []
                abstract_list = []
                nsyl_list = []
                decode_lemma_list = []
                decode_original_list = []

            df_data = pd.concat(dfs_all)
            if len(dfs)>0:
                if n_words>=250:
                    dfs_500.append(pd.concat(dfs))
                else:
                    if len(dfs_500)>0:
                        dfs_500[-1] = pd.concat([dfs_500[-1],pd.concat(dfs)])
                    else:
                        dfs_500.append(pd.concat(dfs))
                
            density_list = []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if len(dfs_500)>0:
                    for df_500 in dfs_500:
                        df_500 = df_500[~df_500['is_stop']]
                        vectorizer = TfidfVectorizer(smooth_idf=True)
                        X = vectorizer.fit_transform([' '.join(x[1]['lemma'].values.astype(str)) for x in df_500.groupby('sent_index')])
                        
                        if len(df_500['lemma'].unique()) > 1:
                            svd_model = TruncatedSVD(n_components=min(X.shape[1]-1,min(len(df_500['lemma'].unique())-1,10)), algorithm='randomized', n_iter=100, random_state=0)
                            svd_model.fit(X)
                            density_list.append(svd_model.explained_variance_ratio_.max())
            density_list = [x for x in density_list if not np.isnan(x)]

            summary = pd.concat([pd.Series({'std_length':np.std(sent_length_list_all),
                                            'high_mean_length': self.high_avg(sent_length_list_all),
                                            'mean_depth':np.nanmean(sent_depth_list),
                                            'std_depth':np.std(sent_depth_list),
                                            'density':np.mean(density_list) if len(density_list) else 0}),
                                sum(summaries)/len(summaries)]).sort_index()
            
            # Mean syllable count
            df_data2 = df_data.copy()
            df_data2['original'] = df_data2['original'].apply(lambda x: x.lower())
            
            nsyl_clean_list = df_data2[df_data2['nsyl'] != 0].drop_duplicates('original')['nsyl'].tolist()
            summary['nsyl_mean'] = np.nanmean(nsyl_clean_list)
            
            # High mean of the syllable count
            summary['nsyl_high_mean'] = self.high_avg(nsyl_clean_list)
            
            # Decoding Demand
            summary['decode_lemma'] = np.nanmean(df_data['decode_lemma'].tolist())
            summary['decode_original'] = np.nanmean(df_data['decode_original'].tolist())
            
            abstract_score_list = df_data['abstract'].tolist()
            
            if len(abstract_score_list) == 0:
                summary['abstract_mean'] = 0
                summary['abstract_high_mean'] = 0
            else:
                summary['abstract_mean'] = np.nanmean(abstract_score_list)
                summary['abstract_high_mean'] = self.high_avg(abstract_score_list)

            # RASCH MODEL ABILIES AT X%
            ability_at = [0.5, 0.7, 0.9]
            
            aoa_clean_list = df_data['aoa'].tolist()
            
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
                    for a in ability_at:
                        a *= int(100)
                        summary[f'aoa_{a}'] = np.percentile(aoa_clean_list, a)
            
            if len(df_data) == 0:
                summary['freq_mean'] = 0
                summary['freq_high_mean'] = 0
                for a in ability_at:
                    summary['freq_' + str(int(a*100))] = 0
            else:
                rareness_clean_list = (1-df_data['frequency']).tolist()
                # summary['freq_mean'] = np.nanmean(rareness_clean_list)
                # summary['freq_high_mean'] = self.high_avg(rareness_clean_list)

                try:
                    freq_ability = self.get_ability_2(rareness_clean_list, ability_at)
                    for a in ability_at:
                        summary['freq_' + str(int(a*100))] = freq_ability[a]
                except:
                    for a in ability_at:
                        a *= int(100)
                        summary[f'freq_{a}'] = np.percentile(rareness_clean_list, a)

            pos_count = df_data['pos'].value_counts().to_dict()
            total = sum([v for k,v in pos_count.items() if k not in {'SPACE','PUNCT'}])
            n_sents = len(df_data.sent_index.unique())
            summary['adj'] = pos_count.get('ADJ',0)/n_sents
            summary['adv'] = pos_count.get('ADV',0)/n_sents
            summary['pron'] = pos_count.get('PRON',0)/n_sents
            summary['n'] = pos_count.get('NOUN',0)/n_sents
            summary['v'] = (pos_count.get('VERB',0)+pos_count.get('PROPN',0))/n_sents
            summary['n_words'] = total
            summary['n_sents'] = n_sents
            summary = summary.fillna(0).infer_objects(copy=False)
            
            return df_data, summary, longest_sent

        def predict(self, summary):
            selected = ['density', 'hdd', 'high_mean_length', 'maas_ttr', 'mean_depth',
                        'mean_lcs2', 'mean_lcs3', 'mean_length', 'mean_lev_distance',
                        'mean_log_freq_token', 'mean_log_freq_token_no_stop',
                        'mean_log_freq_type', 'mean_log_freq_type_no_stop', 'mtld',
                        'noncompressibility', 'std_depth', 'std_length', 'nsyl_mean',
                        'nsyl_high_mean', 'decode_lemma', 'decode_original', 'abstract_mean',
                        'abstract_high_mean', 'aoa_mean', 'aoa_high_mean', 'aoa_50.0',
                        'aoa_70.0', 'aoa_90.0', 'freq_50.0', 'freq_70.0', 'freq_90.0', 'adj',
                        'adv', 'pron', 'n', 'v']
            x = summary[selected].values

            y_pred = neural_model2.predict(x.reshape(1,-1),verbose=0)[0][0]
            if y_pred>1200:
                y_pred = linear_model2.predict([x])[0]
            return y_pred

        def get_age_grade(self, score):
            tree = {
                1: {"age": "6-7", "min": 5, "max": 365},
                2: {"age": "7-8", "min": 245, "max": 605},
                3: {"age": "8-9", "min": 480, "max": 810},
                4: {"age": "9-10", "min": 700, "max": 1005},
                5: {"age": "10-11", "min": 795, "max": 1100},
                6: {"age": "11-12", "min": 875, "max": 1180},
                7: {"age": "12-13", "min": 940, "max": 1250},
                8: {"age": "13-14", "min": 1000, "max": 1310},
                9: {"age": "14-15", "min": 1050, "max": 1360},
                10: {"age": "15-16", "min": 1095, "max": 1410},
                11: {"age": "16-17", "min": 1140, "max": 1450},
                12: {"age": "17-18", "min": 1140, "max": 1450},
            }

            age_arr = []
            grade_arr = []

            for k, v in tree.items():
                if v.get("min") < score < v.get("max"):
                    age_arr.append(v.get("age"))
                    grade_arr.append(k)
            return age_arr, grade_arr

        def decoding_score(self,decode_original, nsyl_mean):
            
            def decode_original_score(decode_original):
                score = decode_original*65.39133259-283.6882894370211
                return min(max(0,score),100)
            
            def nsyl_mean_score(nsyl_mean):
                score = nsyl_mean*99.85680598-115.78580562563019
                return min(max(0,score),100)

            return round((decode_original_score(decode_original)+nsyl_mean_score(nsyl_mean))/2)

        def vocabulary_score(self,aoa_mean, abstract_mean):
            def aoa_mean_score(aoa_mean):
                score = aoa_mean*892.71956214-97.75377982567898
                return min(max(0,score),100)

            def abstract_mean_score(abstract_mean):
                score = abstract_mean*507.3187022066814-216.68550802262394
                return min(max(0,score),100)

            return round((aoa_mean_score(aoa_mean)+abstract_mean_score(abstract_mean))/2)

        def sentences_score(self,mean_length,mean_lev_distance,mean_lcs2,mean_lcs3):
            def mean_length_score(mean_length):
                if mean_length<=10.10957323579106:
                    score = mean_length*6.327764402396921-30.907628775917466
                else:
                    score = mean_length*1.6111335335070158+20.334566597896405
                return min(max(0,score),100)
                
            def mean_lev_distance_score(mean_lev_distance):
                if mean_lev_distance<=35.361661682477596:
                    score = mean_lev_distance*1.1268500723996848-8.888730667142777
                else:
                    score = mean_lev_distance*0.4887343674123663+18.879299145806463
                return min(max(0,score),100)
                
            def mean_lcs2_score(mean_lcs2):
                if mean_lcs2>=0.13108869661396375:
                    score = -mean_lcs2*61.60541610543943+30.508087957115094
                else:
                    score = -mean_lcs2*930.0869518370125+134.71343549058005
                return min(max(0,score),100)

            def mean_lcs3_score(mean_lcs3):
                if mean_lcs3>=0.06996621013558413:
                    score = -mean_lcs3*55.94615995985709+25.675220044474045
                else:
                    score = -mean_lcs3*1834.988672752836+138.92626227564108
                return min(max(0,score),100)

            return round((mean_length_score(mean_length)+mean_lev_distance_score(mean_lev_distance)+mean_lcs2_score(mean_lcs2)+mean_lcs3_score(mean_lcs3))/4)

        def pattern_score(self, mtld, hdd, maas_ttr):
            def mtld_score(mtld):
                if mtld<=39.028965941459376:
                    score = mtld*1.2029939831310108-10.331639374848105
                else:
                    score = mtld*2.022225630558942-59.9106423293764
                return min(max(0,score),100)

            def hdd_score(hdd):
                if hdd<=0.7346020755811701:
                    score = hdd*41.1402686901111-5.870845601305838
                else:
                    score = hdd*953.1471630414901-698.6406673192134
                return min(max(0,score),100)

            def maas_ttr_score(maas_ttr):
                if maas_ttr>=0.062039455292699344:
                    score = -maas_ttr*230.81562229283628+39.401885259002086
                else:
                    score = -maas_ttr*3380.355700350255+220.0842881252814
                return min(max(0,score),100)
                
            return round((mtld_score(mtld)+hdd_score(hdd)+maas_ttr_score(maas_ttr))/3)

        def start_analyze(self):
            df, summary, longest_sentence = self.process_text()

            count_lemmas = df.pivot_table(index=['lemma', 'pos'], aggfunc='size')
            
            word_pos = count_lemmas.index.tolist()
            word_count = count_lemmas.values.tolist()
            word_count = sum(word_count)

            #pred_lexile = predict(summary, neural_model, transformer, kde_model, outlier_ranges)
            pred_lexile = self.predict(summary)

            mean_sent_length = round(word_count/summary['n_sents'],2)

            scores = {
                "catile": round(pred_lexile),
                "decoding": self.decoding_score(summary['decode_original'], summary['nsyl_mean']),
                "vocabulary": self.vocabulary_score(summary['aoa_mean'], summary['abstract_mean']),
                "sentences": self.sentences_score(summary['mean_length'], summary['mean_lev_distance'], summary['mean_lcs2'], summary['mean_lcs3']),
                "patterns": self.pattern_score(summary['mtld'], summary['hdd'], summary['maas_ttr']),
            }

            #return scores

            df_data = df.merge(df.pivot_table(index=['lemma'], aggfunc='size').to_frame(name='lemma_frequency').reset_index(),on='lemma',how='left')
            df_data = df_data[~df_data['is_stop']&df_data['pos'].apply(lambda x: x not in {'X','PROPN','AUX','PART'})].drop_duplicates('lemma')
            df_data = df_data[df_data['lemma'].apply(lambda x:len(x)>=2) & ~((df_data['pos']=='NUM')&(df_data['frequency']<=0))]
            #most_frequent = df_data.sort_values(['lemma_frequency','frequency','aoa','nsyl','decode_lemma'],ascending=[False,True,False,False,False]).head()['lemma'].values
            difficult_words = df_data.sort_values(['frequency','aoa','lemma_frequency','nsyl','decode_lemma'],ascending=[True,False,False,False,False]).head(10)['lemma'].values
            
            age, grade = self.get_age_grade(pred_lexile)

            self.result = {
                "scores": scores,
                "difficult_words": difficult_words, 
                "longest_sentence": longest_sentence,
                "mean_sent_length": mean_sent_length,
                "word_count": word_count,
                "age": age,
                "grade": grade
            }


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
            data = np.array(data)
            data = data[data != 0]

            if str(n_bins) != 'auto':
                h, data_hist = np.histogram(data, bins = n_bins, density=True)  
            else:
                h, data_hist = np.histogram(data, bins = 'auto', density=True)
        
            h_csum = np.cumsum(h)
            h_csum_norm = h_csum / h_csum.max()

            est_x = np.nanmean(data_hist)

            init_vals = [1/est_x,est_x,0]
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
            arr = np.array(arr)
            arr = arr[arr != 0]
            
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
            return c + ((1-c) / (1+np.exp(np.float128(a*(b-x)))))

        def rasch_func_solve(self, x, arr):
            a = arr[0]
            b = arr[1]
            c = arr[2]
            y = arr[3]
            return c + ((1-c) / (1+np.exp(np.float128(a*(b-x))))) - y

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

                    if lemma_list[-1] in most_freq_50 or x.is_stop:
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
            summary = summary.fillna(0).infer_objects(copy=False)
            
            return df_data, summary, longest_sent

        def predict(self, summary):
            x = summary[['density','hdd','high_mean_length','maas_ttr','mean_depth','mean_lcs2','mean_lcs3','mean_length','mean_lev_distance','mean_log_freq_token','mean_log_freq_token_no_stop','mean_log_freq_type',
                        'mean_log_freq_type_no_stop','mtld','noncompressibility','std_depth','std_length','nsyl_mean','nsyl_high_mean','decode_original','decode_lemma','abstract_mean','abstract_high_mean','aoa_mean','aoa_high_mean']].values

            y_pred = neural_model.predict(x.reshape(1,-1),verbose=0)[0][0]
            if y_pred>1200:
                y_pred = linear_model.predict([x])[0]
            return y_pred

        def get_age_grade(self, score):
            tree = {
                1: {"age": "6-7", "min": 5, "max": 365},
                2: {"age": "7-8", "min": 245, "max": 605},
                3: {"age": "8-9", "min": 480, "max": 810},
                4: {"age": "9-10", "min": 700, "max": 1005},
                5: {"age": "10-11", "min": 795, "max": 1100},
                6: {"age": "11-12", "min": 875, "max": 1180},
                7: {"age": "12-13", "min": 940, "max": 1250},
                8: {"age": "13-14", "min": 1000, "max": 1310},
                9: {"age": "14-15", "min": 1050, "max": 1360},
                10: {"age": "15-16", "min": 1095, "max": 1410},
                11: {"age": "16-17", "min": 1140, "max": 1450},
                12: {"age": "17-18", "min": 1140, "max": 1450},
            }

            age_arr = []
            grade_arr = []

            for k, v in tree.items():
                if v.get("min") < score < v.get("max"):
                    age_arr.append(v.get("age"))
                    grade_arr.append(k)
            return age_arr, grade_arr

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
                "catile": round(pred_lexile),
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
            
            age, grade = self.get_age_grade(pred_lexile)

            self.result = {
                "scores": scores,
                "difficult_words": difficult_words, 
                "longest_sentence": longest_sentence,
                "mean_sent_length": mean_sent_length,
                "word_count": word_count,
                "age": age,
                "grade": grade
            }


    class CEFRAnalyzer2(object):

        def __init__(self,outer):
            self.shared_object = outer

            self.settings = {'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'custom_dictionary':{}}
            self.outputs=['final_levels']
            

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
                    ('be done', 'ger.'):'Gerund simple passive',
                    ('be doing', 'inf.'):'Infinitive continuous',
                    ('have done', 'inf.'):'Infinitive perfect',
                    ('have been doing', 'ind. (past)'):'Past perfect continuous',
                    ('have been done', 'ind. (past)'):'Past perfect passive',
                    ('be being done', 'ind. (past)'):'Past continuous passive',
                    ('have been done', 'inf.'):'Infinitive perfect passive',
                    ('have done', 'ger.'):'Gerund perfect',
                    ('have been doing', 'inf.'):'Infinitive perfect continuous',
                    ('have been doing', 'ger.'):'Gerund perfect continuous',
                    ('have been done', 'ger.'):'Gerund perfect passive'}

            self.result = None
            self.embeddings = {}


        def get_individual_word_cefr(self, word):

            def search_word(word, df_dict):
                x = word.lower()
                df_temp = df_dict[df_dict['lemma']==x]
                if len(df_temp)>0:
                    df_temp = df_temp.sort_values('level')
                    row = {'pos':df_temp.iloc[0]['pos'], 'level':df_temp.iloc[0]['level']}
                    return row
                return

            row = None
            for df_dict in wordlist_dicts:
                row = search_word(word,df_dict)
                if row is not None:
                    break
            if row is None:
                row = {'pos':'X', 'level':6}

            return row['pos'], row['level']

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

        def get_phrase(self, phrase, phrase_pos, phrase_dep, sentence, sentence_start_index, followed_by, window_size=3):

            confidence = length = len(phrase)
            ambiguous = False
            sm = edit_distance.SequenceMatcher(a=phrase, b=sentence, action_function=edit_distance.highest_match_action)
            opcodes = sm.get_opcodes()
            filter_ = []
            for opcode in opcodes:
                if opcode[0] == 'replace':
                    if not (phrase_pos[opcode[1]]==self.shared_object.doc[opcode[3]+sentence_start_index].pos_ and phrase_pos[opcode[1]] in set(['PRON','DET'])):
                        return None, 0, ambiguous
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

            if operations_count['equal']+operations_count['replace']!=length or operations_count['equal']<=length/2 or 'delete' in operations_count:
                return None, 0, ambiguous
            
            for i in range(1,len(matching_blocks)):
                word_in_sentence = self.shared_object.doc[matching_blocks[i][1]+sentence_start_index]
                if word_in_sentence.pos_ in set(["ADP","ADV"]) and phrase_dep is not None and phrase_dep[matching_blocks[i][0]]=='prt' and word_in_sentence.dep_!='prt':
                    ambiguous = True

                n_insertions = matching_blocks[i][1]-matching_blocks[i-1][1]-1
                if followed_by[i-1]=="0":
                    if opcodes[matching_blocks[i][1]-1][0] not in set(['equal','replace']):
                        return None, 0, ambiguous
                else:
                    if self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index].pos_=="VERB":
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.tag_ in set(['RP','IN']):
                                return None, 0, ambiguous
                    if followed_by[i-1]=="1":
                        if n_insertions!=1:
                            confidence -= abs(1-n_insertions)
                    elif followed_by[i-1]==",":
                        if n_insertions>1:
                            confidence -= abs(1-n_insertions)
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i][1]-1+sentence_start_index].pos_ != 'PUNCT':
                            return None, 0, ambiguous
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
                                return None, 0, ambiguous
                        if n_insertions==0:
                            confidence -= 1
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i-1][1]+1+sentence_start_index].pos_ == 'DET':
                            return None, 0, ambiguous
                    elif followed_by[i-1]=="s":
                        is_possessive = False
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.pos_ in set(['VERB','AUX']):
                                return None, 0, ambiguous
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
                return None, 0, ambiguous
            
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
                return None, 0, ambiguous

            return span, confidence/len(phrase), ambiguous

        def get_sentence_parts(self, x, followed_by, window_size=3):
            followed_by = followed_by.split('_')[0]
            phrase_length = len(followed_by)+1+followed_by.count('1')+followed_by.count('2')+(followed_by.count('3')+followed_by.count('4'))*window_size
            sentence_parts = []
            start_index = max(x.sent[0].i, x.i-phrase_length)
            end_index = min(x.i+phrase_length, x.sent[-1].i)
            for i in range(start_index,end_index+1):
                if self.shared_object.doc[i].lemma_=='not':
                    sentence_parts.append(['not','never','hardly','barely'])
                elif self.shared_object.doc[i].lemma_ in set(['can','could']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['can','could'])
                elif self.shared_object.doc[i].lemma_ in set(['will','would']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['will','would'])
                elif self.shared_object.doc[i].lemma_ in set(['may','might']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['may','might'])
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
                id_range = sorted(list(set(id_range)))
                if self.shared_object.doc[id_range[0]].tag_=='MD' and self.shared_object.doc[id_range[1]].tag_!='VB':
                    return None, None
                return tense, id_range

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
            #if x.dep_=='xcomp' and list(x.children)[0].lemma_!='to' and list(x.children)[0].pos_!='PART':
            #    return tense, id_range
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
            elif sum([form.startswith(y) for y in set(['do ','does','has','have','am','is','are','gets','get'])])>0:
                tense2 = 'ind. (present)'
            elif form=='do':
                first = x
                while first.dep_ == 'conj':
                    first = first.head
                if all([child.dep_!='nsubj' for child in first.children if child.i<first.i]) and str(x.morph) == 'VerbForm=Inf' and not (x.i>=4 and self.shared_object.doc[x.i-1].lemma_=='but' and self.shared_object.doc[x.i-2].lemma_=='help' and self.shared_object.doc[x.i-3].lemma_=='not' and self.shared_object.doc[x.i-4].lemma_ in ['can','could']):
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
            if level is None:
                subtree = None
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
                if clause_form == 'like':
                    level = 2
                elif clause_form.endswith('ever'):
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

        def float2cefr(self, num):
            cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
            output = cefr.get(int(num),"Native")
            if num<6:
                s = "%.1f" % num
                output += s[s.index('.'):]
            return output

        def float2ielts(self, num):
            if num >= 5.45:
                return 9.0
            elif num >= 5.25:
                return 8.5
            elif num >= 5.0:
                return 8.0
            elif num >= 4.55:
                return 7.5
            elif num >= 4.25:
                return 7.0
            elif num >= 3.8:
                return 6.5
            elif num >= 3.45:
                return 6.0
            elif num >= 3.1:
                return 5.5
            elif num >= 2.7:
                return 5.0
            elif num >= 2.35:
                return 4.5
            elif num >= 2.0:
                return 4.0
            else:
                return round(num * 2 / 0.5) * 0.5

        def score2grades(self, score):
            ket = "KET "
            if 120 <= score <= 132:
                ket += "Grade C"
            elif 133 <= score <= 139:
                ket += "Grade B"
            elif 140 <= score <= 150:
                ket += "Grade A"
            else:
                ket = ""

            pet = "PET "
            if 140 <= score <= 152:
                pet += "Grade C"
            elif 153 <= score <= 159:
                pet += "Grade B"
            elif 160 <= score <= 170:
                pet += "Grade A"
            else:
                pet = ""

            fce = "FCE "
            if 160 <= score <= 172:
                fce += "Grade C"
            elif 173 <= score <= 179:
                fce += "Grade B"
            elif 180 <= score <= 190:
                fce += "Grade A"
            else:
                fce = ""

            cae = "CAE "
            if 180 <= score <= 192:
                cae += "Grade C"
            elif 193 <= score <= 199:
                cae += "Grade B"
            elif 200 <= score <= 210:
                cae += "Grade A"
            else:
                cae = ""

            cpe = "CPE "
            if 200 <= score <= 212:
                cpe += "Grade C"
            elif 213 <= score <= 219:
                cpe += "Grade B"
            elif 220 <= score <= 230:
                cpe += "Grade A"
            else:
                cpe = ""

            return [x for x in [ket, pet, fce, cae, cpe] if x]


        def float2exams(self, num):
            score = int(num*20+100)
            grades = self.score2grades(score)
            ielts = self.float2ielts(num)
            return {'cambridge_scale_score':score,'exam_grades':grades,'ielts':ielts}

        
        def sentence_length_level(self, length):
            a,b = [0.20722492, 3.30842234]
            return 6/(1+np.exp(-length*a+b))
        
        def cefr2length(self, level):
            a,b = [0.20722492, 3.30842234]
            level = max(0,min(level,5.9))
            return round(max(1,(b-np.log(6/level-1))/a)) if level else 1

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
                temp_dict = df_reference_words[df_reference_words['level_0']==stem].to_dict('records')
                for row in temp_dict:
                    try:
                        levels.append(np.argmax(cefr_word_model.predict(self.process_input(*row.values,word,pos).reshape(1,-1))))
                    except:
                        pass
            if len(levels)==0:
                for stem in stems:
                    temp_dict = df_reference_words_sup[df_reference_words_sup['level_0']==stem].to_dict('records')
                    for row in temp_dict:
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
            
        def run_embed(self):
            sentences = []
            for k,v in self.embeddings.items():
                if v is None:
                    sentences.append(v['sentence'])
            embeddings = sentence_model.encode(sentences)
            for i in range(len(embeddings)):
                self.embeddings[sentences[i]] = embeddings[i]
            
        def embed_all(self, dfs):
            sentences = []
            em_indeces = {}
            head_indeces = {}
            for k,df in dfs.items():
                sentence = df.to_dict('list')
                tokens = []
                for i in range(len(sentence['word'])):
                    tokens.append(sentence['word'][i] + ' '*sentence['whitespace'][i])
                for i in range(len(tokens)):
                    if sentence['pos'][i]!= 'PUNCT' and sentence['pos'][i]!= 'SPACE' and sentence['lemma'][i] not in most_freq_50:
                        words = tokens+[]
                        words[i] = "[UNK]"+' '*sentence['whitespace'][i]
                        s = ''.join(words).strip()
                        if len(s.split(' '))>1:
                            em_indeces[f'{k}_{i}'] = len(sentences)
                            sentences.append(''.join(words).strip())
                head_indeces[k] = len(sentences)
                sentences.append(''.join(tokens).strip())
            ems = sentence_model.encode(sentences)
            X = []
            for k,df in dfs.items():
                self.embeddings[sentences[head_indeces[k]]] = ems[head_indeces[k]]
                weights = [0]*len(df)
                for i in range(len(df)):
                    if f'{k}_{i}' in em_indeces:
                        weights[i] = np.linalg.norm(ems[em_indeces[f'{k}_{i}']]-ems[head_indeces[k]])
                weights = np.array(weights)
                sum_weights = sum(weights)
                dfs[k]['weight'] = weights/sum_weights if sum_weights else weights
                X.append(self.extract_sentence_features(df)+list(ems[head_indeces[k]]))
            self.features = X
            
            CEFR_sentences = {}
            pred = np.round(sentence_level_model.predict(X,verbose = 0),1)
            for i,k in enumerate(dfs.keys()):
                CEFR_sentences[k] = min(max(0,pred[i][0]),6)
            self.CEFR_sentences = CEFR_sentences

            return dfs

        def extract_sentence_features(self, df):
            sentences = df.to_dict('list')
            pos = np.array(sentences['pos'])
            if 'CEFR_vocabulary' not in sentences:
                sentences['CEFR_vocabulary'] = sentences['CEFR']
            features = [
                len([x for x in pos if x not in ['PUNCT','SPACE']]),
                sentences['pos'].count("PRON"),
                sentences['pos'].count("NOUN")+sentences['pos'].count("PROPN"),
                sentences['pos'].count("VERB"),
                sentences['pos'].count("ADV"),
                sentences['pos'].count("ADJ"),
                sentences['CEFR_vocabulary'].count(-1)+sentences['CEFR_vocabulary'].count(0),
                sentences['CEFR_vocabulary'].count(1),
                sentences['CEFR_vocabulary'].count(2)+sentences['CEFR_vocabulary'].count(3),
                sentences['CEFR_vocabulary'].count(4)+sentences['CEFR_vocabulary'].count(5)+sentences['CEFR_vocabulary'].count(6),
            ]

            features.append(sum((np.maximum(sentences['CEFR_vocabulary'],0)+1)*np.array(sentences['weight']))-1)

            if len(sentences['CEFR_tense'])>0:
                features.append(max(np.nan_to_num(sentences['CEFR_tense'])))
            else:
                features.append(0)
                
            clause_count,_ = self.count_clause_span(df)
            features.append(clause_count[0])
            features.append(clause_count[1])
            features.append(clause_count[2]+clause_count[3])
            features.append(clause_count[4]+clause_count[5])
            return features

        def get_word_weights(self, sentence):
            sentences = []
            em_indeces = {}
            tokens = []
            for i in range(len(sentence['word'])):
                tokens.append(sentence['word'][i] + ' '*sentence['whitespace'][i])

            for i in range(len(tokens)):
                if sentence['pos'][i]!= 'PUNCT' and sentence['pos'][i]!= 'SPACE' and sentence['lemma'][i] not in most_freq_50:
                    words = tokens+[]
                    words[i] = "[UNK]"+' '*sentence['whitespace'][i]
                    sentences.append(''.join(words))
                    em_indeces[i] = len(sentences)-1
            sentences.append(''.join(tokens))
            weights = [0]*len(tokens)
            ems = sentence_model.encode(sentences)
            for i in range(len(tokens)):
                if i in em_indeces:
                    weights[i] = np.linalg.norm(ems[em_indeces[i]]-ems[-1])
            weights = np.array(weights)
            sum_weights = sum(weights)
            return weights/sum_weights if sum_weights else weights


        def tag_text(self, sentences):
            num2cefr = {-1:'CEFR_A0',0:'CEFR_A1',1:'CEFR_A2',2:'CEFR_B1',3:'CEFR_B2',4:'CEFR_C1',5:'CEFR_C2',6:'CEFR_D'}
            text_tagged = ''
            for _, s in sentences.items():
                for i in range(len(s['pos'])):
                    try:
                        cefr = num2cefr.get(s['CEFR_vocabulary'][i])
                    except:
                        cefr = num2cefr.get(s['CEFR'][i])
                    if cefr:
                        text_tagged += f"<{cefr}>" + s['word'][i] + f"</{cefr}>" + ' '*s['whitespace'][i]
                    else:
                        text_tagged += s['word'][i] + ' '*s['whitespace'][i]
            return text_tagged


        def print_time(self, message):
            #print(message, time.time()-self.t0)
            self.t0 = time.time()

        def process(self):
            dfs = {}
            rows = []

            verb_form_time = 0
            clause_time = 0
            vocabulary_time = 0
            phrase_time = 0
            typing_time = 0
            self.t0 = time.time()

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
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True, 'phrase_is_idiom':True})
                    elif not bool(re.match(".*[A-Za-z]+",x.lemma_)):
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':'PUNCT','CEFR':-2,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                    'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True, 'phrase_is_idiom':True})
                    elif self.settings['as_wordlist']:
                        pos, level = self.get_individual_word_cefr(x.orth_)
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.orth_.lower(),'pos':pos,'CEFR':level,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                    'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True, 'phrase_is_idiom':True})
                    
                    else:
                        skip = False
                        if x.pos_ == 'INTJ' and self.settings['intj_as_lowest']==True:
                            skip = True
                        elif x.pos_ == 'PROPN':
                            if self.settings['propn_as_lowest']==True:
                                skip = True
                            else:
                                x.lemma_ = lemmatizer.lemmatize(x.lemma_.lower())
                            
                        if not skip:
                            #x = fine_lemmatize(x,self.shared_object.doc,nlp)

                            tense_level = None
                            form = None
                            tense_span = None
                            tense1 = None
                            tense2 = None
                            tense_term = None
                            
                            
                            self.t0 = time.time()
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
                            verb_form_time += time.time()-self.t0
                                    
                            self.t0 = time.time()
                            # Clauses
                            clause_form, clause, clause_span, clause_level = self.get_clause(x)
                            clause_time += time.time()-self.t0

                            self.t0 = time.time()
                            # Vocabulary
                            if x.orth_.lower() in most_freq_50:
                                word_lemma = tuple([x.lemma_,'STOP'])
                                word_orth = tuple([x.orth_.lower(),'STOP'])
                            else:
                                word_lemma = tuple([x.lemma_,x.pos_])
                                word_orth = tuple([x.orth_.lower(),x.pos_])

                            if self.settings['keep_min']:
                                cefr_w_pos_prim = cefr_w_pos_min_prim
                                cefr_wo_pos_prim = cefr_wo_pos_min_prim
                            else:
                                cefr_w_pos_prim = cefr_w_pos_mean_prim
                                cefr_wo_pos_prim = cefr_wo_pos_mean_prim

                            level = None
                            if len(self.settings['custom_dictionary'])>0:
                                for key in [word_lemma,word_orth,x.lemma_,x.orth_.lower(),x.orth_]:
                                    level = self.settings['custom_dictionary'].get(key,None)
                                    if level is not None:
                                        break
                            if level is None:
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
                        vocabulary_time += time.time()-self.t0

                        self.t0 = time.time()
                        # Phrases
                        phrase = None
                        phrase_span = None
                        max_confidence = 0
                        ambiguous = True
                        is_idiom = True
                        
                        if 'phrase_count' in self.outputs:
                            phrases_words = set(df_phrases['word'].values)
                            if x.pos_ not in set(["DET","PART"]) and x.lemma_.lower() in phrases_words:
                                #max_phrase_length = 0
                                max_clean_length = 0

                                df_phrases_temp = df_phrases[df_phrases['word']==x.lemma_]
                                sentence_parts = []
                                phrases_dict_temp = df_phrases_temp.to_dict('records')
                                for row in phrases_dict_temp:

                                    #if phrase is not None and phrase.startswith(row['original']) and max_confidence==1:
                                    #    continue

                                    phrase_parts = row['phrase_parts']
                                    phrase_length = len(phrase_parts)
                                    #if phrase_length > max_phrase_length:
                                    #    sentence_parts, start_index = self.get_sentence_parts(x,phrase_length)
                                    #    max_phrase_length = phrase_length

                                    sentence_parts, start_index = self.get_sentence_parts(x,row['followed_by'])
                                    
                                    if phrase_length>len(sentence_parts) or len(set(sum(sentence_parts,[])).intersection(set(row['lemma'])))<phrase_length:
                                        continue
                                    phrase_span_temp, confidence_temp, prt_ambiguous = self.get_phrase(phrase_parts, row['pos'], row['dep'], sentence_parts, start_index, row['followed_by'])
                                    if (phrase_span_temp is not None and confidence_temp>0 and (confidence_temp>max_confidence or 
                                                                                                confidence_temp==max_confidence and (phrase_length>max_clean_length or 
                                                                                                                                    confidence_temp==1 and row['pos'][-1]!='ADP' and phrase_length==max_clean_length))):
                                        phrase_span = list(np.array(phrase_span_temp) + start_index)
                                        phrase = row['original_to_display']
                                        max_clean_length = phrase_length*1
                                        max_confidence = confidence_temp*1
                                        ambiguous = row['ambiguous'] or prt_ambiguous
                                        is_idiom = row['is_idiom']
                        phrase_time += time.time()-self.t0

                        if skip:
                            rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':-1,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                        'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                        'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                        'phrase':phrase, 'phrase_span':phrase_span,'phrase_confidence':max_confidence,'phrase_ambiguous':ambiguous, 'phrase_is_idiom':is_idiom})
                        else:
                            rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':level,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                        'form':form,'tense1':tense1,'tense2':tense2,'tense_term':tense_term,'CEFR_tense':tense_level,'tense_span':tense_span,
                                        'clause_form':clause_form,'clause':clause,'CEFR_clause':clause_level,'clause_span':clause_span,
                                        'phrase':phrase, 'phrase_span':phrase_span,'phrase_confidence':max_confidence,'phrase_ambiguous':ambiguous, 'phrase_is_idiom':is_idiom})

                self.t0 = time.time()
                df_lemma = pd.DataFrame(rows)

                if len(rows)>0 and len(df_lemma[df_lemma['CEFR']>=-1])>0:
                    df_lemma['id'] = df_lemma['id'].astype(int)
                    df_lemma['word'] = df_lemma['word'].astype(str)
                    df_lemma['lemma'] = df_lemma['lemma'].astype(str)
                    df_lemma['pos'] = df_lemma['pos'].astype(str)
                    df_lemma['CEFR'] = df_lemma['CEFR'].astype(float)
                    df_lemma['whitespace'] = df_lemma['whitespace'].astype(bool)
                    df_lemma['sentence_id'] = df_lemma['sentence_id'].astype(int)
                    df_lemma['form'] = df_lemma['form'].astype(object)
                    df_lemma['tense1'] = df_lemma['tense1'].astype(object)
                    df_lemma['tense2'] = df_lemma['tense2'].astype(object)
                    df_lemma['tense_term'] = df_lemma['tense_term'].astype(object)
                    df_lemma['CEFR_tense'] = df_lemma['CEFR_tense'].astype(float)
                    df_lemma['tense_span'] = df_lemma['tense_span'].astype(object)
                    df_lemma['clause_form'] = df_lemma['clause_form'].astype(object)
                    df_lemma['clause'] = df_lemma['clause'].astype(object)
                    df_lemma['CEFR_clause'] = df_lemma['CEFR_clause'].astype(float)
                    df_lemma['clause_span'] = df_lemma['clause_span'].astype(object)
                    df_lemma['phrase'] = df_lemma['phrase'].astype(str)
                    df_lemma['phrase_span'] = df_lemma['phrase_span'].astype(object)
                    df_lemma['phrase_confidence'] = df_lemma['phrase_confidence'].astype(float)
                    df_lemma['phrase_ambiguous'] = df_lemma['phrase_ambiguous'].astype(bool)
                    df_lemma['phrase_is_idiom'] = df_lemma['phrase_is_idiom'].astype(bool)
                    dfs[n_sent] = df_lemma
                    rows = []
                typing_time += time.time()-self.t0
            

            # print('verb_form_time',verb_form_time)
            # print('clause_time',clause_time)
            # print('vocabulary_time',vocabulary_time)
            # print('phrase_time',phrase_time)
            # print('typing_time',typing_time)
            

            self.t0 = time.time()
            if len(dfs)>0:
                dfs = self.embed_all(dfs)
                df_lemma = pd.concat(dfs.values())
            else:
                df_lemma = pd.DataFrame([],columns=['id','word','lemma','pos','CEFR','whitespace','weight','sentence_id',
                                    'form','tense1','tense2','tense_term','CEFR_tense','tense_span',
                                    'clause_form','clause','CEFR_clause','clause_span',
                                    'phrase', 'phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom'])
            self.print_time('embed_all')
            self.df_lemma = df_lemma

            if len(self.outputs)==0:
                self.result = {}
                return

            n_words = len(df_lemma[(df_lemma['pos']!='PUNCT')&(df_lemma['pos']!='SPACE')])
            
            n_clausal = 0
            n_clauses = 0
            sentence_levels = []
            sentences = {}

            dfs_phrase_count = []

            if 'wordlists' in self.outputs:
                topic_vocabulary = {}
                pos_temp = {'NOUN','VERB','ADJ','ADV'}
                df_lemma_temp = df_lemma[df_lemma.apply(lambda x: x['pos'] in pos_temp and x['lemma'] not in most_freq_50 and len(x['lemma'])>1, axis = 1)].copy()
                df_lemma_temp['lemma_pos'] = df_lemma_temp['lemma'] + '_' + df_lemma_temp['pos']
                topic_vocabulary = {topic:{} for topic in topic_words.keys()}
                for topic, subtopics in topic_words.items():
                    for subtopic, words in subtopics.items():
                        temp = list(set(df_lemma_temp['lemma_pos']).intersection(words))
                        if len(temp)>0:
                            topic_vocabulary[topic][subtopic] = temp
            self.print_time('Topic vocabulary')

            sentence_levels = []
            sentence_lengths = []
            clause_levels = []
            
            for sentence_id, df in dfs.items():
                length = len(df[(df['pos']!='PUNCT')&(df['pos']!='SPACE')])
                sentence_lengths.append(length)
                sentence_levels += [self.CEFR_sentences[sentence_id]]*length

                df_clause = df[df['CEFR_clause'].fillna(0)>0]
                # if len(df_clause)>0:
                #     level_by_clause = min(sum(df_clause['CEFR_clause']*np.minimum((df_clause['clause_span'].apply(lambda x: max(1,max(x)-min(x)))/length/0.5),1)),6)
                # else:
                #     level_by_clause = 0
                # clause_level = max(np.mean([level_by_length,level_by_clause]),level_by_clause)
                # clause_levels.append(clause_level)
                levels_by_clause = sum((df_clause['CEFR_clause'].apply(lambda x:[x])*df_clause['clause_span'].apply(lambda x: max(1,max(x)-min(x)))).values,[])
                levels_by_clause += [0]*max(0,(length-len(levels_by_clause)))
                clause_levels += levels_by_clause

                if 'phrase_count' in self.outputs:
                    df2 = df[['phrase','phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom','sentence_id']].dropna()
                    if len(df2)>0:
                        filter_ = []
                        spans = df2['phrase_span'].values
                        for i in range(len(spans)):
                            unique = True
                            for j in range(len(spans)):
                                if i!=j:
                                    if spans[i][0]>=spans[j][0] and spans[i][-1]<=spans[j][-1]:
                                        if not(len(spans[i])==len(spans[j]) and spans[i][0]==spans[j][0] and spans[i][-1]<spans[j][-1]):
                                            unique = False
                                            break
                                    elif len(spans[i])==len(spans[j]) and spans[i][0]==spans[j][0] and spans[i][-1]>spans[j][-1]:
                                        unique = False
                                        break
                            filter_.append(unique)
                        df2 = df2[filter_]
                        dfs_phrase_count.append(df2)
                        phrase_dict = df2[['phrase','phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom']].to_dict(orient='list')
                    else:
                        phrase_dict = {'phrase':[],'phrase_span':[],'phrase_confidence':[],'phrase_ambiguous':[],'phrase_is_idiom':[]}

                if 'sentences' in self.outputs or 'clause_stats' in self.outputs:
                    lemma_dict = df[['id','word','lemma','pos','whitespace','CEFR','weight']].to_dict(orient='list')
                    lemma_dict['CEFR_vocabulary'] = lemma_dict['CEFR']+[]
                    del lemma_dict['CEFR']
                    df2 = df[['form','tense1','tense2','CEFR_tense','tense_span']].dropna()
                    #level_dict = {'CEFR_vocabulary':6,'CEFR_tense':5}
                    level_dict = {}
                    if len(df2)>0:
                        tense_dict = df2[['form','tense1','tense2','CEFR_tense','tense_span']].to_dict(orient='list')
                        #level_dict['CEFR_tense'] = max(tense_dict['CEFR_tense'])
                    else:
                        tense_dict = {'form':[],'tense1':[],'tense2':[],'CEFR_tense':[],'tense_span':[]}
                        #level_dict['CEFR_tense'] = 0

                    df2 = df[['clause_form','clause','clause_span','CEFR_clause']].dropna()
                    if len(df2)>0:
                        n_clausal += 1
                        n_clauses += len(df2)

                        df2['span_string'] = df2['clause_span'].astype(str)
                        df2['span_string2'] = df2['clause'].astype(str)
                        clause_dict = df2.drop_duplicates(['span_string','span_string2'])[['clause_form','clause','clause_span','CEFR_clause']].to_dict(orient='list')
                    else:
                        clause_dict = {'clause_form':[],'clause':[],'CEFR_clause':[],'clause_span':[]}

                    #_,cumsum_series = self.sum_cumsum(df[df['CEFR']>=-1])

                    #level_dict['CEFR_vocabulary'] = self.estimate_95(df[df['CEFR']>=-1]['CEFR'].values)
                    #level_dict['CEFR_clause'] = round(clause_level,1)
                    
                    if 'phrase_count' in self.outputs:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict,**phrase_dict,'CEFR_sentence':self.CEFR_sentences[sentence_id]}
                    else:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict,'CEFR_sentence':self.CEFR_sentences[sentence_id]}
                
            self.print_time('sentences')

            if 'wordlists' in self.outputs or 'modified_final_levels' in self.outputs:
                wordlists = {}
                for CEFR,group in df_lemma[df_lemma['CEFR']>=-1].groupby('CEFR'):
                    df = group[['lemma','pos']]
                    wordlists[CEFR] = df.groupby(df.columns.tolist(),as_index=False).size().sort_values(['size','lemma','pos'],ascending=[False,True,True]).to_dict(orient='list')
                
            self.print_time('wordlists')

            if 'tense_count' in self.outputs or 'tense_stats' in self.outputs:
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

            self.print_time('tense_count')

            if 'tense_term_count' in self.outputs or 'tense_stats' in self.outputs:
                tense_term_count = {}

                df_lemma_temp = df_lemma[~pd.isnull(df_lemma['form'])&~pd.isnull(df_lemma['tense_term'])&~pd.isnull(df_lemma['tense1'])&~pd.isnull(df_lemma['tense2'])].copy().reset_index(drop=True)
                
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
            self.print_time('tense_term_count')

            sum_tense, cumsum_tense = self.count_tense(df_lemma)
            tense_stats = {'sum_token':{'values':list(sum_tense.astype(int))},'cumsum_token':{'values':list(np.round(cumsum_tense.values,4))}}
            a,b,c,d = self.fit_sigmoid(cumsum_tense.values,np.arange(0,5.5,0.5))
            tense_stats['cumsum_token']['constants']=[a,b,c,d]
            values = []
            for k,v in sum_tense.to_dict().items():
                values += [k]*int(v)
            tense_stats['level'] = {'fit_curve':[self.percentile2level(0.95,a,b,c,d)],
                                    #'ninety_five':[self.ninety_five(cumsum_tense,5)],
                                    'ninety_five':[self.estimate_95(values,minimum=0,maximum=5,step=0.5)],
                                    'fit_error':[self.fit_error(cumsum_tense.values[1:],np.arange(0.5,5.5,0.5),a,b,c,d)]}

            self.print_time('tense_stats')

            if 'clause_count' in self.outputs:
                clause_count = {}
                for clause,group in df_lemma[~pd.isnull(df_lemma['clause_form'])&~pd.isnull(df_lemma['clause'])].groupby('clause'):
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
            self.print_time('clause_count')
            

            if len(clause_levels)>0:
                mean_length = round(np.mean(sentence_lengths),1)
                length_level = self.sentence_length_level(mean_length)
                clause_level = np.percentile(clause_levels,95)

                if clause_level<length_level:
                    clause_level = np.mean([clause_level,length_level])
                clause_level = round(min(clause_level,6),1)
            else:
                clause_level = 0
                mean_length = 0

            if 'clause_stats' in self.outputs:
                sum_clause, cumsum_clause = self.count_clause(df_lemma)
                clause_stats = {'sum_token':{'values':list(sum_clause.astype(int))},'cumsum_token':{'values':list(np.round(cumsum_clause.values,4))}}
                # a,b,c,d = self.fit_sigmoid(cumsum_clause.values,np.arange(0,6,1))
                # clause_stats['cumsum_token']['constants']=[a,b,c,d]
                values = []
                for k,v in sum_clause.to_dict().items():
                    values += [k]*int(v)

                mean_clause = n_clausal and n_clauses/n_clausal or 0
                clause_stats['level'] = {#'fit_curve':[self.percentile2level(0.95,a,b,c,d)],
                                        'ninety_five':[np.percentile(clause_levels,95)],
                                        #'fit_error':[self.fit_error(cumsum_clause.values[1:],np.arange(1,6,1),a,b,c,d)]
                                        }
            
                clause_stats.update({'p_clausal':len(dfs) and n_clausal/len(dfs) or 0,'mean_clause':mean_clause,'mean_length':mean_length,'length_level':length_level,'n_words':n_words})

            self.print_time('clause_stats')

            if 'phrase_count' in self.outputs:
                phrase_count = {}
                if len(dfs_phrase_count)>0:
                    df_phrase_count = pd.concat(dfs_phrase_count)
                    for phrase,group in df_phrase_count.groupby('phrase',as_index=True):
                        group['span_string'] = group['phrase_span'].astype(str)
                        group = group.drop_duplicates(['span_string','sentence_id'])
                        temp_df = group.agg(len)['sentence_id']
                        temp_dict = {'id':phrase_original2id.get(phrase,0),'phrase_ambiguous':any(group['phrase_ambiguous'].tolist()),'phrase_is_idiom':group['phrase_is_idiom'].tolist()[0],'size':temp_df.tolist(),'phrase_span':group['phrase_span'].tolist(),'phrase_confidence':group['phrase_confidence'].tolist(),'sentence_id':group['sentence_id'].astype(int).tolist()}
                        phrase_count[phrase] = temp_dict
            self.print_time('phrase_count')

            sum_series_token, cumsum_series_token, sum_series_type, cumsum_series_type = self.count_cefr(df_lemma)
            
            stats_dict = {'sum_token':{'values':list(sum_series_token.astype(int))},
                        'cumsum_token':{'values':list(np.round(cumsum_series_token.values,4))},
                        'sum_type':{'values':list(sum_series_type.astype(int))},
                        'cumsum_type':{'values':list(np.round(cumsum_series_type.values,4))}}
            self.print_time('clause_stats2')


            for k in set(['cumsum_token','cumsum_type']):
                a,b,c,d = self.fit_sigmoid((stats_dict[k]['values'][1:-1]),range(6))
                stats_dict[k]['constants']=[a,b,c,d]
                if k == 'cumsum_token':
                    stats_dict['level'] = {'fit_curve':[self.percentile2level(0.95,a,b,c,d)],
                                        #'ninety_five':[self.ninety_five(cumsum_series_token)],
                                        'ninety_five':[self.estimate_95(df_lemma[df_lemma['CEFR']>=-1]['CEFR'].values)],
                                        'fit_error':[self.fit_error(stats_dict[k]['values'][2:-1],range(1,6),a,b,c,d)]}
                    
            self.print_time('stats')

            if 'final_levels' in self.outputs or 'tense_stats' in self.outputs or 'modified_final_levels' in self.outputs:
                if tense_stats["level"]["fit_error"][0]>=0.05:
                    tense_level = tense_stats["level"]["ninety_five"][0]
                else:
                    tense_level = tense_stats["level"]["fit_curve"][0]
                # if clause_stats["level"]["fit_error"][0]>=0.1:
                #     clause_level = clause_stats["level"]["ninety_five"][0]
                # else:
                #     clause_level = clause_stats["level"]["fit_curve"][0]
                if stats_dict["level"]["fit_error"][0]>=0.1:
                    vocabulary_level = stats_dict["level"]["ninety_five"][0]
                else:
                    vocabulary_level = stats_dict["level"]["fit_curve"][0]

                sentence_level = np.percentile(sentence_levels,80)
                
                general_level = (max([vocabulary_level,tense_level,clause_level])+sentence_level)/2

                final_levels = {'general_level':max(0,min(round(general_level,1),6)),
                                'vocabulary_level':max(0,min(round(vocabulary_level,1),6)),
                                'tense_level':max(0,min(round(tense_level,1),6)),
                                'clause_level':max(0,min(round(clause_level,1),6)),
                                'sentence_level':max(0,min(round(sentence_level,1),6))}
            self.print_time('final_levels')

            if 'modified_final_levels' in self.outputs:
                df_lemma_temp2 = df_lemma.copy()
                buttom_level = max(tense_level,clause_level-0.5)
                wordlist_dfs = []
                ignored_words = []
                modified_final_levels = []
                if final_levels['vocabulary_level']>buttom_level and int(final_levels['vocabulary_level'])>0:
                    for l in reversed(range(6)):
                        if l<=buttom_level:
                            break
                        wordlist = wordlists.get(l)
                        if wordlist:
                            wordlist = pd.DataFrame(wordlist)
                            wordlist_dfs.append(wordlist[wordlist['size']>=3])
                    if len(wordlist_dfs)>0:
                        difficult_words = pd.concat(wordlist_dfs).sort_values('size').to_dict('records')
                        for x in difficult_words:
                            for i,row in enumerate(df_lemma_temp2.to_dict('records')):
                                if x['lemma']==row['lemma'] and x['pos']==row['pos']:
                                    df_lemma_temp2.iat[i,4] = -1
                            ignored_words.append(x['lemma']+"_"+x['pos'])
                            modified_vocabulary_level = self.estimate_95(df_lemma_temp2['CEFR'].values)
                            modified_general_level = (max([modified_vocabulary_level,tense_level,clause_level])+sentence_level)/2
                            temp = {'ignored_words':ignored_words+[],'final_levels':{'general_level':min(round(modified_general_level,1),6),
                                                                                                        'vocabulary_level':min(round(modified_vocabulary_level,1),6),
                                                                                                        'tense_level':min(round(tense_level,1),6),
                                                                                                        'clause_level':min(clause_level,6)}}
                            temp['final_levels_str'] = {k:self.float2cefr(v) for k,v in temp['final_levels'].items()}
                            temp['exam_stats'] = self.float2exams(temp['final_levels']['general_level'])
                            modified_final_levels.append(temp)
                            if modified_vocabulary_level<=min(buttom_level,int(buttom_level)+0.5):
                                break

            self.print_time('modified_final_levels')

            result_dict = {}
            if 'sentences' in self.outputs:
                result_dict['sentences'] = sentences
                result_dict['text_tagged'] = self.tag_text(sentences)
                #self.sentences = sentences
            if 'wordlists' in self.outputs:
                result_dict['wordlists'] = wordlists
                result_dict['topic_vocabulary'] = topic_vocabulary
                #self.wordlists = wordlists
            if 'vocabulary_stats' in self.outputs:
                result_dict['stats'] = stats_dict
                #self.vocabulary_stats = stats_dict
            if 'tense_count' in self.outputs:
                result_dict['tense_count'] = tense_count
                #self.tense_count = tense_count
            if 'tense_term_count' in self.outputs:
                result_dict['tense_term_count'] = tense_term_count
            if 'tense_stats' in self.outputs:
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
            if 'clause_count' in self.outputs:
                result_dict['clause_count'] = clause_count
                #self.clause_count = clause_count
            if 'clause_stats' in self.outputs:
                result_dict['clause_stats'] = clause_stats
                #self.clause_stats = clause_stats
            if 'phrase_count' in self.outputs:
                result_dict['phrase_count'] = phrase_count
                #self.clause_count = clause_count
            if 'final_levels' in self.outputs:
                result_dict['final_levels'] = final_levels
                result_dict['final_levels_str'] = {k:self.float2cefr(v) for k,v in final_levels.items()}
                result_dict['exam_stats'] = self.float2exams(final_levels['general_level'])
            if 'modified_final_levels' in self.outputs:
                result_dict['modified_final_levels'] = modified_final_levels
                #self.final_levels = final_levels
            self.print_time('outputs')
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
        
        def count_clause_span(self,df_lemma):
            base = pd.Series(dict(zip(np.arange(0,6,1),[0.]*6)))
            max_clause = {}
            for row in df_lemma.to_dict('records'):
                if not pd.isnull(row['CEFR_clause']):
                    for i in row['clause_span']:
                        max_clause[i] = max(max_clause.get(i,0),row['CEFR_clause'])
                elif row['id'] not in max_clause and row['pos']!='PUNCT' and row['pos']!='SPACE':
                    max_clause[row['id']] = 0

            counts = base.add(pd.Series(Counter(max_clause.values())),fill_value=0.)
            if counts.sum()==0:
                p = base
            else:
                p = counts/counts.sum()
            return counts, p.cumsum()

        def count_clause(self,df_lemma):
            base = pd.Series(dict(zip(np.arange(0,6,1),[0.]*6)))
            counts = base.add(df_lemma[~pd.isnull(df_lemma['CEFR_clause'])].groupby('CEFR_clause')['word'].agg(len),fill_value=0.)
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
            return max(0,a/(-b+np.exp(-x*c+d)))

        def level2percentile(self,y,a,b,c,d):
            y = np.array(y)
            return (d-np.log(a/y+b))/c

        def estimate_95(self, levels, minimum=-1, maximum=6, step=1):
            try:
                cumsum = np.cumsum([Counter(levels).get(i,0) for i in np.arange(minimum,maximum+step,step)])
                if cumsum[-1]==0:
                    return 0
                cumsum = cumsum/cumsum[-1]
                for i in range(len(cumsum)-1):
                    if cumsum[i]>=0.95:
                        return max(0,i*step+minimum)
                    elif cumsum[i+1]>0.95:
                        return max(0,(i+(0.95-cumsum[i])/(cumsum[i+1]-cumsum[i]))*step+minimum)
                return maximum
            except:
                return 0

        def ninety_five(self,cumsum_series,default=6):
            if cumsum_series.sum()==0:
                return 0
            level = default
            try:
                for i,v in cumsum_series.items():
                    if v>=0.95:
                        level = i
                        break
            except:
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

        def start_analyze(self):
            self.process()


    class CEFRAnalyzer(object):

        def __init__(self,outer):
            self.shared_object = outer

            self.__settings = {'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,
                    'return_sentences':True, 'return_wordlists':True,'return_vocabulary_stats':True,
                    'return_tense_count':True,'return_tense_term_count':True,'return_tense_stats':True,'return_clause_count':True,
                    'return_clause_stats':True,'return_phrase_count':True,'return_final_levels':True,'return_modified_final_levels':False}

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
                    ('be done', 'ger.'):'Gerund simple passive',
                    ('be doing', 'inf.'):'Infinitive continuous',
                    ('have done', 'inf.'):'Infinitive perfect',
                    ('have been doing', 'ind. (past)'):'Past perfect continuous',
                    ('have been done', 'ind. (past)'):'Past perfect passive',
                    ('be being done', 'ind. (past)'):'Past continuous passive',
                    ('have been done', 'inf.'):'Infinitive perfect passive',
                    ('have done', 'ger.'):'Gerund perfect',
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

        def get_phrase(self, phrase, phrase_pos, phrase_dep, sentence, sentence_start_index, followed_by, window_size=3):

            confidence = length = len(phrase)
            ambiguous = False
            sm = edit_distance.SequenceMatcher(a=phrase, b=sentence, action_function=edit_distance.highest_match_action)
            opcodes = sm.get_opcodes()
            filter_ = []
            for opcode in opcodes:
                if opcode[0] == 'replace':
                    if not (phrase_pos[opcode[1]]==self.shared_object.doc[opcode[3]+sentence_start_index].pos_ and phrase_pos[opcode[1]] in set(['PRON','DET'])):
                        return None, 0, ambiguous
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

            if operations_count['equal']+operations_count['replace']!=length or operations_count['equal']<=length/2 or 'delete' in operations_count:
                return None, 0, ambiguous
            
            for i in range(1,len(matching_blocks)):
                word_in_sentence = self.shared_object.doc[matching_blocks[i][1]+sentence_start_index]
                if word_in_sentence.pos_ in set(["ADP","ADV"]) and phrase_dep is not None and phrase_dep[matching_blocks[i][0]]=='prt' and word_in_sentence.dep_!='prt':
                    ambiguous = True

                n_insertions = matching_blocks[i][1]-matching_blocks[i-1][1]-1
                if followed_by[i-1]=="0":
                    if opcodes[matching_blocks[i][1]-1][0] not in set(['equal','replace']):
                        return None, 0, ambiguous
                else:
                    if self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index].pos_=="VERB":
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.tag_ in set(['RP','IN']):
                                return None, 0, ambiguous
                    if followed_by[i-1]=="1":
                        if n_insertions!=1:
                            confidence -= abs(1-n_insertions)
                    elif followed_by[i-1]==",":
                        if n_insertions>1:
                            confidence -= abs(1-n_insertions)
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i][1]-1+sentence_start_index].pos_ != 'PUNCT':
                            return None, 0, ambiguous
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
                                return None, 0, ambiguous
                        if n_insertions==0:
                            confidence -= 1
                        elif n_insertions==1 and self.shared_object.doc[matching_blocks[i-1][1]+1+sentence_start_index].pos_ == 'DET':
                            return None, 0, ambiguous
                    elif followed_by[i-1]=="s":
                        is_possessive = False
                        for x in self.shared_object.doc[matching_blocks[i-1][1]+sentence_start_index+1:matching_blocks[i][1]+sentence_start_index]:
                            if x.pos_ in set(['VERB','AUX']):
                                return None, 0, ambiguous
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
                return None, 0, ambiguous
            
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
                return None, 0, ambiguous

            return span, confidence/len(phrase), ambiguous

        def get_sentence_parts(self, x, followed_by, window_size=3):
            followed_by = followed_by.split('_')[0]
            phrase_length = len(followed_by)+1+followed_by.count('1')+followed_by.count('2')+(followed_by.count('3')+followed_by.count('4'))*window_size
            sentence_parts = []
            start_index = max(x.sent[0].i, x.i-phrase_length)
            end_index = min(x.i+phrase_length, x.sent[-1].i)
            for i in range(start_index,end_index+1):
                if self.shared_object.doc[i].lemma_=='not':
                    sentence_parts.append(['not','never','hardly','barely'])
                elif self.shared_object.doc[i].lemma_ in set(['can','could']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['can','could'])
                elif self.shared_object.doc[i].lemma_ in set(['will','would']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['will','would'])
                elif self.shared_object.doc[i].lemma_ in set(['may','might']) and self.shared_object.doc[i].pos_=='AUX':
                    sentence_parts.append(['may','might'])
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
                id_range = sorted(list(set(id_range)))
                if self.shared_object.doc[id_range[0]].tag_=='MD' and self.shared_object.doc[id_range[1]].tag_!='VB':
                    return None, None
                return tense, id_range

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
            #if x.dep_=='xcomp' and list(x.children)[0].lemma_!='to' and list(x.children)[0].pos_!='PART':
            #    return tense, id_range
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
            elif sum([form.startswith(y) for y in set(['do ','does','has','have','am','is','are','gets','get'])])>0:
                tense2 = 'ind. (present)'
            elif form=='do':
                first = x
                while first.dep_ == 'conj':
                    first = first.head
                if all([child.dep_!='nsubj' for child in first.children if child.i<first.i]) and str(x.morph) == 'VerbForm=Inf' and not (x.i>=4 and self.shared_object.doc[x.i-1].lemma_=='but' and self.shared_object.doc[x.i-2].lemma_=='help' and self.shared_object.doc[x.i-3].lemma_=='not' and self.shared_object.doc[x.i-4].lemma_ in ['can','could']):
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
                if clause_form == 'like':
                    level = 2
                elif clause_form.endswith('ever'):
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

        def float2cefr(self, num):
            cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
            output = cefr.get(int(num),"Native")
            if num<6:
                s = "%.1f" % num
                output += s[s.index('.'):]
            return output

        def float2ielts(self, num):
            if num >= 5.45:
                return 9.0
            elif num >= 5.25:
                return 8.5
            elif num >= 5.0:
                return 8.0
            elif num >= 4.55:
                return 7.5
            elif num >= 4.25:
                return 7.0
            elif num >= 3.8:
                return 6.5
            elif num >= 3.45:
                return 6.0
            elif num >= 3.1:
                return 5.5
            elif num >= 2.7:
                return 5.0
            elif num >= 2.35:
                return 4.5
            elif num >= 2.0:
                return 4.0
            else:
                return round(num * 2 / 0.5) * 0.5

        def score2grades(self, score):
            ket = "KET "
            if 120 <= score <= 132:
                ket += "Grade C"
            elif 133 <= score <= 139:
                ket += "Grade B"
            elif 140 <= score <= 150:
                ket += "Grade A"
            else:
                ket = ""

            pet = "PET "
            if 140 <= score <= 152:
                pet += "Grade C"
            elif 153 <= score <= 159:
                pet += "Grade B"
            elif 160 <= score <= 170:
                pet += "Grade A"
            else:
                pet = ""

            fce = "FCE "
            if 160 <= score <= 172:
                fce += "Grade C"
            elif 173 <= score <= 179:
                fce += "Grade B"
            elif 180 <= score <= 190:
                fce += "Grade A"
            else:
                fce = ""

            cae = "CAE "
            if 180 <= score <= 192:
                cae += "Grade C"
            elif 193 <= score <= 199:
                cae += "Grade B"
            elif 200 <= score <= 210:
                cae += "Grade A"
            else:
                cae = ""

            cpe = "CPE "
            if 200 <= score <= 212:
                cpe += "Grade C"
            elif 213 <= score <= 219:
                cpe += "Grade B"
            elif 220 <= score <= 230:
                cpe += "Grade A"
            else:
                cpe = ""

            return [x for x in [ket, pet, fce, cae, cpe] if x]


        def float2exams(self, num):
            score = int(num*20+100)
            grades = self.score2grades(score)
            ielts = self.float2ielts(num)
            return {'cambridge_scale_score':score,'exam_grades':grades,'ielts':ielts}

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
                temp_dict = df_reference_words[df_reference_words['level_0']==stem].to_dict('records')
                for row in temp_dict:
                    try:
                        levels.append(np.argmax(cefr_word_model.predict(self.process_input(*row.values,word,pos).reshape(1,-1))))
                    except:
                        pass
            if len(levels)==0:
                for stem in stems:
                    temp_dict = df_reference_words_sup[df_reference_words_sup['level_0']==stem].to_dict('records')
                    for row in temp_dict:
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


        def tag_text(self, sentences):
            num2cefr = {-1:'CEFR_A0',0:'CEFR_A1',1:'CEFR_A2',2:'CEFR_B1',3:'CEFR_B2',4:'CEFR_C1',5:'CEFR_C2',6:'CEFR_D'}
            text_tagged = ''
            for _, s in sentences.items():
                for i in range(len(s['pos'])):
                    try:
                        cefr = num2cefr.get(s['CEFR_vocabulary'][i])
                    except:
                        cefr = num2cefr.get(s['CEFR'][i])
                    if cefr:
                        text_tagged += f"<{cefr}>" + s['word'][i] + f"</{cefr}>" + ' '*s['whitespace'][i]
                    else:
                        text_tagged += s['word'][i] + ' '*s['whitespace'][i]
            return text_tagged

        def process(self, custom_dictionary={}):
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
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True, 'phrase_is_idiom':True})
                    elif not bool(re.match(".*[A-Za-z]+",x.lemma_)):
                        rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':'PUNCT','CEFR':-2,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                    'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                    'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                    'phrase':None, 'phrase_span':None,'phrase_confidence':None, 'phrase_ambiguous':True, 'phrase_is_idiom':True})
                    else:
                        skip = False
                        if x.pos_ == 'INTJ' and self.__settings['intj_as_lowest']==True:
                            skip = True
                        elif x.pos_ == 'PROPN':
                            if self.__settings['propn_as_lowest']==True:
                                skip = True
                            else:
                                x.lemma_ = lemmatizer.lemmatize(x.lemma_.lower())
                            
                        if not skip:
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
                            if x.orth_.lower() in most_freq_50:
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

                            level = None
                            if len(custom_dictionary)>0:
                                for key in [word_lemma,word_orth,x.lemma_,x.orth_.lower(),x.orth_]:
                                    level = custom_dictionary.get(key,None)
                                    if level is not None:
                                        break
                            if level is None:
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
                        is_idiom = True

                        if self.__settings['return_phrase_count']:
                            phrases_words = set(df_phrases['word'].values)
                            if x.pos_ not in set(["DET","PART"]) and x.lemma_.lower() in phrases_words:
                                #max_phrase_length = 0
                                max_clean_length = 0

                                df_phrases_temp = df_phrases[df_phrases['word']==x.lemma_]
                                sentence_parts = []
                                phrases_dict_temp = df_phrases_temp.to_dict('records')
                                for row in phrases_dict_temp:

                                    #if phrase is not None and phrase.startswith(row['original']) and max_confidence==1:
                                    #    continue

                                    phrase_parts = row['phrase_parts']
                                    phrase_length = len(phrase_parts)
                                    #if phrase_length > max_phrase_length:
                                    #    sentence_parts, start_index = self.get_sentence_parts(x,phrase_length)
                                    #    max_phrase_length = phrase_length

                                    sentence_parts, start_index = self.get_sentence_parts(x,row['followed_by'])
                                    
                                    if phrase_length>len(sentence_parts) or len(set(sum(sentence_parts,[])).intersection(set(row['lemma'])))<phrase_length:
                                        continue
                                    phrase_span_temp, confidence_temp, prt_ambiguous = self.get_phrase(phrase_parts, row['pos'], row['dep'], sentence_parts, start_index, row['followed_by'])
                                    if (phrase_span_temp is not None and confidence_temp>0 and (confidence_temp>max_confidence or 
                                                                                                confidence_temp==max_confidence and (phrase_length>max_clean_length or 
                                                                                                                                    confidence_temp==1 and row['pos'][-1]!='ADP' and phrase_length==max_clean_length))):
                                        phrase_span = list(np.array(phrase_span_temp) + start_index)
                                        phrase = row['original_to_display']
                                        max_clean_length = phrase_length*1
                                        max_confidence = confidence_temp*1
                                        ambiguous = row['ambiguous'] or prt_ambiguous
                                        is_idiom = row['is_idiom']
                                    
                        if skip:
                            rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':-1,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                        'form':None,'tense1':None,'tense2':None,'CEFR_tense':None,'tense_span':None,'tense_term':None,
                                        'clause_form':None,'clause':None,'CEFR_clause':None,'clause_span':None,
                                        'phrase':phrase, 'phrase_span':phrase_span,'phrase_confidence':max_confidence,'phrase_ambiguous':ambiguous, 'phrase_is_idiom':is_idiom})
                        else:
                            rows.append({'id':x.i,'word':x.orth_,'lemma':x.lemma_,'pos':x.pos_,'CEFR':level,'whitespace':bool(is_white_space),'sentence_id':n_sent,
                                        'form':form,'tense1':tense1,'tense2':tense2,'tense_term':tense_term,'CEFR_tense':tense_level,'tense_span':tense_span,
                                        'clause_form':clause_form,'clause':clause,'CEFR_clause':clause_level,'clause_span':clause_span,
                                        'phrase':phrase, 'phrase_span':phrase_span,'phrase_confidence':max_confidence,'phrase_ambiguous':ambiguous, 'phrase_is_idiom':is_idiom})

                df_lemma = pd.DataFrame(rows)
                if len(rows)>0 and len(df_lemma[df_lemma['CEFR']>=-1])>0:
                    dfs[n_sent] = df_lemma
                    rows = []

            if len(dfs)>0:
                df_lemma = pd.concat(dfs.values())
                df_lemma['id'] = df_lemma['id'].astype(int)
                df_lemma['word'] = df_lemma['word'].astype(str)
                df_lemma['lemma'] = df_lemma['lemma'].astype(str)
                df_lemma['pos'] = df_lemma['pos'].astype(str)
                df_lemma['CEFR'] = df_lemma['CEFR'].astype(float)
                df_lemma['whitespace'] = df_lemma['whitespace'].astype(bool)
                df_lemma['sentence_id'] = df_lemma['sentence_id'].astype(int)
                df_lemma['form'] = df_lemma['form'].astype(object)
                df_lemma['tense1'] = df_lemma['tense1'].astype(object)
                df_lemma['tense2'] = df_lemma['tense2'].astype(object)
                df_lemma['tense_term'] = df_lemma['tense_term'].astype(object)
                df_lemma['CEFR_tense'] = df_lemma['CEFR_tense'].astype(float)
                df_lemma['tense_span'] = df_lemma['tense_span'].astype(object)
                df_lemma['clause_form'] = df_lemma['clause_form'].astype(object)
                df_lemma['clause'] = df_lemma['clause'].astype(object)
                df_lemma['CEFR_clause'] = df_lemma['CEFR_clause'].astype(float)
                df_lemma['clause_span'] = df_lemma['clause_span'].astype(object)
                df_lemma['phrase'] = df_lemma['phrase'].astype(str)
                df_lemma['phrase_span'] = df_lemma['phrase_span'].astype(object)
                df_lemma['phrase_confidence'] = df_lemma['phrase_confidence'].astype(float)
                df_lemma['phrase_ambiguous'] = df_lemma['phrase_ambiguous'].astype(bool)
                df_lemma['phrase_is_idiom'] = df_lemma['phrase_is_idiom'].astype(bool)
            else:
                df_lemma = pd.DataFrame([],columns=['id','word','lemma','pos','CEFR','whitespace','sentence_id',
                                    'form','tense1','tense2','tense_term','CEFR_tense','tense_span',
                                    'clause_form','clause','CEFR_clause','clause_span',
                                    'phrase', 'phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom'])
            self.df_lemma = df_lemma
            n_words = len(df_lemma[(df_lemma['pos']!='PUNCT')&(df_lemma['pos']!='SPACE')])
            
            n_clausal = 0
            n_clauses = 0
            clause_levels = []
            sentences = {}

            dfs_phrase_count = []

            if self.__settings['return_sentences'] or self.__settings['return_wordlists'] or self.__settings['return_modified_final_levels']:
                # dfs_keyterms = []
                # for pos in ['NOUN','VERB','ADJ','ADV']:
                #     df_keyterms = pd.DataFrame(extract.keyterms.yake(self.shared_object.doc, ngrams=1, include_pos=(pos,),topn=1.), columns=['lemma','yake'])
                #     df_keyterms['pos'] = pos
                #     dfs_keyterms.append(df_keyterms)
                # df_keyterms = pd.concat(dfs_keyterms)
                # df_lemma = df_lemma.merge(df_keyterms,how='left',on=['pos','lemma'])
                # df_lemma['yake'] = df_lemma['yake'].fillna(1)
                # self.df_lemma = df_lemma

                topic_vocabulary = {}
                pos_temp = set(['NOUN','VERB','ADJ','ADV'])
                df_lemma_temp = df_lemma[df_lemma.apply(lambda x: x['pos'] in pos_temp and x['lemma'] not in most_freq_50 and len(x['lemma'])>1, axis = 1)].copy()
                df_lemma_temp['lemma_pos'] = df_lemma_temp['lemma'] + '_' + df_lemma_temp['pos']
                topic_vocabulary = {topic:{} for topic in topic_words.keys()}
                for topic, subtopics in topic_words.items():
                    for subtopic, words in subtopics.items():
                        temp = list(set(df_lemma_temp['lemma_pos']).intersection(words))
                        if len(temp)>0:
                            topic_vocabulary[topic][subtopic] = temp


            for sentence_id, df in dfs.items():
                # if self.__settings['return_sentences'] or self.__settings['return_wordlists']:
                #     df = df.merge(df_keyterms,how='left',on=['pos','lemma'])
                #     df['yake'] = df['yake'].fillna(1)

                total_span = max(1,len(set(sum(df['clause_span'].dropna().values,[]))))
                level_by_clause = max(max(df['CEFR_clause'].fillna(0)),sum(df['CEFR_clause'].fillna(0).values*df['clause_span'].fillna('').apply(len).values)/total_span)
                level_by_length = min(max(0,1.1**len(df[(df['pos']!='PUNCT')&(df['pos']!='SPACE')])-1.5),7)
                clause_level = min(np.nanmean([level_by_length,level_by_clause]),6)
                clause_levels.append(clause_level)

                if self.__settings['return_phrase_count']:
                    df2 = df[['phrase','phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom','sentence_id']].dropna()
                    if len(df2)>0:
                        filter_ = []
                        spans = df2['phrase_span'].values
                        for i in range(len(spans)):
                            unique = True
                            for j in range(len(spans)):
                                if i!=j:
                                    if spans[i][0]>=spans[j][0] and spans[i][-1]<=spans[j][-1]:
                                        if not(len(spans[i])==len(spans[j]) and spans[i][0]==spans[j][0] and spans[i][-1]<spans[j][-1]):
                                            unique = False
                                            break
                                    elif len(spans[i])==len(spans[j]) and spans[i][0]==spans[j][0] and spans[i][-1]>spans[j][-1]:
                                        unique = False
                                        break
                            filter_.append(unique)
                        df2 = df2[filter_]
                        dfs_phrase_count.append(df2)
                        phrase_dict = df2[['phrase','phrase_span','phrase_confidence','phrase_ambiguous','phrase_is_idiom']].to_dict(orient='list')
                    else:
                        phrase_dict = {'phrase':[],'phrase_span':[],'phrase_confidence':[],'phrase_ambiguous':[],'phrase_is_idiom':[]}

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

                    #_,cumsum_series = self.sum_cumsum(df[df['CEFR']>=-1])

                    level_dict['CEFR_vocabulary'] = self.estimate_95(df[df['CEFR']>=-1]['CEFR'].values)
                    level_dict['CEFR_clause'] = round(clause_level,1)

                    if self.__settings['return_phrase_count']:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict,**phrase_dict}
                    else:
                        sentences[sentence_id] = {**lemma_dict,**tense_dict,**clause_dict,**level_dict}
                
            if self.__settings['return_wordlists'] or self.__settings['return_modified_final_levels']:
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
                for clause,group in df_lemma[~pd.isnull(df_lemma['clause_form'])&~pd.isnull(df_lemma['clause'])].groupby('clause'):
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
                        temp_dict = {'id':phrase_original2id.get(phrase,0),'phrase_ambiguous':any(group['phrase_ambiguous'].tolist()),'phrase_is_idiom':group['phrase_is_idiom'].tolist()[0],'size':temp_df.tolist(),'phrase_span':group['phrase_span'].tolist(),'phrase_confidence':group['phrase_confidence'].tolist(),'sentence_id':group['sentence_id'].astype(int).tolist()}
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
                                        #'ninety_five':[self.ninety_five(cumsum_series_token)],
                                        'ninety_five':[self.estimate_95(df_lemma[df_lemma['CEFR']>=-1]['CEFR'].values)],
                                        'fit_error':[self.fit_error(stats_dict[k]['values'][2:-1],range(1,6),a,b,c,d)]}

            if self.__settings['return_final_levels'] or self.__settings['return_tense_stats'] or self.__settings['return_modified_final_levels']:
                if tense_stats["level"]["fit_error"][0]>=0.05:
                    tense_level = tense_stats["level"]["ninety_five"][0]
                else:
                    tense_level = tense_stats["level"]["fit_curve"][0]

                if stats_dict["level"]["fit_error"][0]>=0.1:
                    vocabulary_level = stats_dict["level"]["ninety_five"][0]
                else:
                    vocabulary_level = stats_dict["level"]["fit_curve"][0]
                
                average_level = (vocabulary_level+tense_level+clause_level)/3
                general_level = max([vocabulary_level,tense_level,average_level,clause_level-0.5])

                final_levels = {'general_level':min(round(general_level,1),6),
                                'vocabulary_level':min(round(vocabulary_level,1),6),
                                'tense_level':min(round(tense_level,1),6),
                                'clause_level':min(clause_level,6)}

            if self.__settings['return_modified_final_levels']:
                df_lemma_temp2 = df_lemma.copy()
                buttom_level = max(tense_level,clause_level-0.5)
                wordlist_dfs = []
                ignored_words = []
                modified_final_levels = []
                if final_levels['vocabulary_level']>buttom_level and int(final_levels['vocabulary_level'])>0:
                    for l in reversed(range(6)):
                        if l<=buttom_level:
                            break
                        wordlist = wordlists.get(l)
                        if wordlist:
                            wordlist = pd.DataFrame(wordlist)
                            wordlist_dfs.append(wordlist[wordlist['size']>=3])
                    if len(wordlist_dfs)>0:
                        difficult_words = pd.concat(wordlist_dfs).sort_values('size').to_dict('records')
                        for x in difficult_words:
                            for i,row in enumerate(df_lemma_temp2.to_dict('records')):
                                if x['lemma']==row['lemma'] and x['pos']==row['pos']:
                                    df_lemma_temp2.iat[i,4] = -1
                            ignored_words.append(x['lemma']+"_"+x['pos'])
                            modified_vocabulary_level = self.estimate_95(df_lemma_temp2['CEFR'].values)
                            modified_average_level = (modified_vocabulary_level+tense_level+clause_level)/3
                            modified_general_level = max([modified_vocabulary_level,tense_level,modified_average_level,clause_level-0.5])
                            temp = {'ignored_words':ignored_words+[],'final_levels':{'general_level':min(round(modified_general_level,1),6),
                                                                                                        'vocabulary_level':min(round(modified_vocabulary_level,1),6),
                                                                                                        'tense_level':min(round(tense_level,1),6),
                                                                                                        'clause_level':min(clause_level,6)}}
                            temp['final_levels_str'] = {k:self.float2cefr(v) for k,v in temp['final_levels'].items()}
                            temp['exam_stats'] = self.float2exams(temp['final_levels']['general_level'])
                            modified_final_levels.append(temp)
                            if modified_vocabulary_level<=min(buttom_level,int(buttom_level)+0.5):
                                break


            result_dict = {}
            if self.__settings['return_sentences']:
                result_dict['sentences'] = sentences
                result_dict['text_tagged'] = self.tag_text(sentences)
                #self.sentences = sentences
            if self.__settings['return_wordlists']:
                result_dict['wordlists'] = wordlists
                result_dict['topic_vocabulary'] = topic_vocabulary
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
                result_dict['final_levels_str'] = {k:self.float2cefr(v) for k,v in final_levels.items()}
                result_dict['exam_stats'] = self.float2exams(final_levels['general_level'])
            if self.__settings['return_modified_final_levels']:
                result_dict['modified_final_levels'] = modified_final_levels
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
            return max(0,a/(-b+np.exp(-x*c+d)))

        def level2percentile(self,y,a,b,c,d):
            y = np.array(y)
            return (d-np.log(a/y+b))/c

        def estimate_95(self, levels, minimum=-1, maximum=6):
            try:
                cumsum = np.cumsum([Counter(levels).get(i,0) for i in range(minimum,maximum+1)])
                cumsum = cumsum/cumsum[-1]
                for i in range(len(cumsum)-1):
                    if cumsum[i]>=0.95:
                        return max(0,i+minimum)
                    elif cumsum[i+1]>0.95:
                        return max(0,i+minimum+(0.95-cumsum[i])/(cumsum[i+1]-cumsum[i]))
                return maximum
            except:
                return 0

        def ninety_five(self,cumsum_series,default=6):
            if cumsum_series.sum()==0:
                return 0
            level = default
            try:
                for i,v in cumsum_series.items():
                    if v>=0.95:
                        level = i
                        break
            except:
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

        def start_analyze(self, propn_as_lowest=True,intj_as_lowest=True,keep_min=True,custom_dictionary={},
                        return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                        return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                        return_clause_stats=True,return_phrase_count=True,return_final_levels=True,return_modified_final_levels=False):
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
            self.__settings['return_modified_final_levels']=return_modified_final_levels
            self.__settings['return_phrase_count']=return_phrase_count
            self.process(custom_dictionary=custom_dictionary)

        def print_settings(self):
            return self.__settings

    class ReadabilityAnalyzer(object):
        def __init__(self,outer):
            self.shared_object = outer
            self.result = None
            self.return_grades=False

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
                result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'fernandez_huerta':textstat.fernandez_huerta(text),
                        'szigriszt_pazos':textstat.szigriszt_pazos(text),
                        'gutierrez_polini':textstat.gutierrez_polini(text),
                        'crawford':textstat.crawford(text)}
            elif language=="it":
                result = {'flesch_reading_ease':textstat.flesch_reading_ease(text),
                        'gulpease_index':textstat.gulpease_index(text)}
            elif language=="pl":
                result = {'gunning_fog':textstat.gunning_fog(text)}
            elif language in set(['de','fr','nl','ru']):
                result = {'flesch_reading_ease':textstat.flesch_reading_ease(text)}
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
                        'mcalpine_eflaw':textstat.mcalpine_eflaw(text)}
                if result['dale_chall_readability_score']<5:
                    result['spache_readability'] = textstat.spache_readability(text)
                result['readability_consensus'] = (result["flesch_kincaid_grade"]+result["gunning_fog"]+result["smog_index"]+result["automated_readability_index"]+result["coleman_liau_index"]+result["linsear_write_formula"]+(result["dale_chall_readability_score"]*2-5))/7
            

            if self.return_grades:
                grades = {k:general_float2grade(v) for k,v in result.items()}
                if 'dale_chall_readability_score' in result:
                    grades['dale_chall_readability_score'] = dale_chall_float2grade(result['dale_chall_readability_score'])
                if 'spache_readability' in result:
                    grades['spache_readability'] = spache_float2grade(result['spache_readability'])
                result['lexicon_count'] = textstat.lexicon_count(text, removepunct=True)
                result['sentence_count'] = textstat.sentence_count(text)
                self.result = {'scores':result,'grades':grades}
            else:
                result['lexicon_count'] = textstat.lexicon_count(text, removepunct=True)
                result['sentence_count'] = textstat.sentence_count(text)
                self.result = result


class AdoVideoAnalyzer(object):
    def __init__(self, text_analyser, temp_dir='temp', openai_api_key=None):
        self.analyser = text_analyser
        self.model = None
        self.temp_dir = temp_dir
        self.openai_api_key = openai_api_key
        self.client = None

    def load_model(self):
        from faster_whisper import WhisperModel
        if torch.cuda.is_available():
            self.model = WhisperModel('medium.en', device="cuda", compute_type="float16")
        else:
            self.model = WhisperModel('medium.en', device="cpu", compute_type="int8")

    def get_video_info(self, url, verbose=False, save_as=None, allow_playlist=False):
        def parse(info_dict):
            text = None
            lines = None
            duration = None
            all_subtitles = info_dict.get('subtitles')
            if all_subtitles is not None:
                en_subtitles = None
                for k,v in all_subtitles.items():
                    if k.startswith('en'):
                        en_subtitles = v
                if en_subtitles is not None:
                    try:
                        lines, duration = self.download_subtitles(en_subtitles)
                        if len(lines)>0:
                            text = ' '.join([x['text'] for x in lines])
                    except:
                        pass
            video_id = info_dict.get('id')
            return {
                'video_id':video_id,
                'title':info_dict.get('title'),
                'url':info_dict.get('webpage_url'),
                'upload_date':info_dict.get('upload_date'),
                'duration':info_dict.get('duration'),
                'channel_id':info_dict.get('channel_id'),
                'channel':info_dict.get('channel'),
                'uploader_id':info_dict.get('uploader_id'),
                'age_limit':info_dict.get('age_limit'),
                'categories':info_dict.get('categories'),
                'text':text,
                'subtitles':lines,
                'speak_duration':duration}

        if allow_playlist==False and not ('v=' in url or 'youtu.be' in url):
            if 'list=' in url:
                raise InformError("Playlist is not supported.")
            else:
                raise InformError("The link is not supported. Please make sure it is a valid YouTube video link.")
        
        ydl_opts = {'subtitleslangs':True, 'noplaylist':True, 'outtmpl': self.temp_dir.strip('\\')+'/'+''.join([str(np.random.randint(0,9)) for i in range(10)])}
        if verbose!=True:
            ydl_opts['logger'] = YoutubeLogger()

        n_trials = 5
        while n_trials>0:
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=False)
                break
            except Exception as e:
                if '404' in str(e) or 'Unsupported' in str(e):
                    raise InformError("The link is not supported. Please make sure it is a valid YouTube video link.")
                elif '403' in str(e):
                    raise InformError("Access to the link is forbidden. Please make sure you have the permission to access the video.")
                n_trials -= 1
                if n_trials == 0:
                    raise Exception(e)

        if 'entries' in info_dict:
            # if save==True:
            #     with open(self.temp_dir.strip('\\')+'/info_dict.pkl', 'wb') as f:
            #         pickle.dump(list(info_dict['entries']), f)
            results = []
            length = len(info_dict['entries'])
            for i,x in enumerate(info_dict['entries']):
                if verbose==True:
                    print(f"Downloading subtitles {i+1} of {length} ...")
                try:
                    results.append(parse(x))
                except:
                    continue
            if save_as:
                with open(self.temp_dir.strip('\\')+f'/{save_as}_info.pkl', 'wb') as f:
                    pickle.dump(results, f)
            return results
        else:
            result = parse(info_dict)
            if save_as:
                with open(self.temp_dir.strip('\\')+f'/{save_as}_info.pkl', 'w') as f:
                    pickle.dump(result, f)
            return result



    def download_subtitles(self,subtitles):
        for x in subtitles:
            if x['ext']=='json3':
                n_trials = 5
                while n_trials>0:
                    try:
                        r = requests.get(x['url'])
                        break
                    except Exception as e:
                        n_trials -= 1
                        if n_trials == 0:
                            raise Exception(e)
                break
        lines = []
        duration = 0
        for x in r.json()['events']:
            line = []
            for y in x['segs']:
                line += [v for v in y.values() if isinstance(v, str)]
            if len(line)==0:
                continue
            line = ' '.join(line)
            new_line = []
            for y in line.split('\n'):
                if re.search(r"^.+(\s.+){0,2}:", y):
                    new_line.append(re.sub(r"^.+(\s.+){0,2}:", "", line))
                else:
                    new_line.append(y)
            line = ' '.join(new_line).replace('\ufeff','').replace('\xa0','')
            line = re.sub(r"\([^()]*\)", "", line)
            line = re.sub(r"\[[^()]*\]", "", line).strip(' ')
            line = re.sub(r'\s+', ' ', line)
            if line=='' or re.sub('[^A-Za-z0-9 ]+', '', line)=='' or 'dDurationMs' not in x:
                continue
            duration += x['dDurationMs']
            lines.append({'start':x['tStartMs']/1000,'end':(x['tStartMs']+x['dDurationMs'])/1000,'text':line})
        return lines, duration/1000


    def transcribe_video(self, url, video_id=None):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        elif self.client is None:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(120, connect=5))

        if video_id is None:
            filename = '%(id)s.mp3'
        else:
            filename = video_id+'.mp3'

        info_dict = {}
        try:
            audio_file = open(self.temp_dir.strip('\\')+'/'+filename, "rb")
        except:
            print("Downloading video ...")
            ydl_opts = {
                'format': 'bestaudio',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'logger':YoutubeLogger(),
                'outtmpl': self.temp_dir.strip('\\')+'/'+filename
            }

            n_trials = 5
            while n_trials>0:
                try:
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                    break
                except Exception as e:
                    n_trials -= 1
                    if n_trials == 0:
                        raise Exception(e)

            if video_id is None:
                filename = info_dict['id']+'.mp3'

            audio_file = open(self.temp_dir.strip('\\')+'/'+filename, "rb")
            
        print("Transcribing ...")
        transcript = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            prompt="Every end of sentence should have a full stop. Only transcribe speech, and don't include music or descriptive captions."
        )
        transcription = ''
        lines = []
        speak_duration = 0
        for x in transcript.segments:
            line = x['text'].strip(' ')
            if line=='' or line.lower()=='music':
                continue
            # if line[-1].isalpha():
            #     line += '.'
            transcription += x['text'] + ' '
            lines.append({'start':x['start'],'end':x['end'],'text':line})
            speak_duration += x['end']-x['start']
        return {'url':url, 'text':transcription, 'subtitles':lines, 'speak_duration':speak_duration}

    def transcribe_audio(self, file_path):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        elif self.client is None:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(120, connect=5))

        audio_file = open(file_path, "rb")
        transcript = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        transcription = ''
        lines = []
        speak_duration = 0
        for x in transcript.segments:
            line = x['text'].strip(' ')
            if line=='' or line.lower()=='music':
                continue
            if line[-1].isalpha():
                line += '.'
            transcription += x['text'] + ' '
            lines.append({'start':x['start'],'end':x['end'],'text':line})
            speak_duration += x['end']-x['start']
        return {'text':transcription, 'subtitles':lines, 'speak_duration':speak_duration}

    def spm_level(self, spm):
        # a,b = [  0.07729107, -14.62657981]
        # return spm*a+b
        a, b = [31.05142857, 89.51238095]
        return min(max(0,(spm-b)/a),6)

    def calculate(self, vocabulary_level,tense_level,clause_level,spm):
        coef = np.array([0.41907501, 0.46284061, 0.28858665, 0.43187369])
        return round(sum(np.array([vocabulary_level,tense_level,clause_level,spm])*coef),1)

    def analyze_audio(self, subtitles,
                      settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}},
                      outputs=['final_levels']):
        
        spms = []
        texts = []
        for i in range(len(subtitles)):
            spms.append(solar_word.count_syllables(subtitles[i]['text'])/max(0.01,subtitles[i]['end']-subtitles[i]['start'])*60)
            texts.append(subtitles[i]['text'].strip(' '))

        result = self.analyser.analyze_cefr(' '.join(texts), settings=settings, outputs=outputs, v=2)
        final_levels = result['final_levels']

        spm = np.median(spms)
        speech_rate_level = self.spm_level(spm)
        final_levels['speech_level'] = round(speech_rate_level,1)
        if speech_rate_level>final_levels['general_level']:
            final_levels['general_level'] = round((final_levels['general_level']+final_levels['speech_level'])/2,1)
        result['speech_stats'] = {'syllable_per_minute':spm, 'speech_rate_level':speech_rate_level}
        result['final_levels'] = final_levels
        result['final_levels_str'] = {k:self.analyser.cefr.float2cefr(v) for k,v in final_levels.items()}
        result['exam_stats'] = self.analyser.cefr.float2exams(final_levels['general_level'])

        return result
    
    def analyze_youtube_video(self, url, transcribe=False, auto_transcribe=True, verbose=False, save_as=None, duration_limit=900,
                              settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}},
                              outputs=['final_levels']):
        print('Getting video info')
        infos = self.get_video_info(url, verbose=verbose)

        if type(infos)!=list:
            infos = [infos]
        n = len(infos)

        if verbose==False:
            print(f'Analysing {n} videos')
        results = []
        for i, x in enumerate(infos):
            x['transcribed'] = False
            if verbose==True:
                print(f'Analysing video {i+1}/{n}')

            if x.get('duration',0)>duration_limit:
                raise InformError(f"This video is too long. Please choose a video that is less than {round(duration_limit/60,1)} minutes long.")

            if not transcribe:
                if x['text'] is None:
                    if not auto_transcribe:
                        results.append({'video_info':x,'result':{'error':'No subtitles found. Analysing videos without English subtitles is not supported yet.'}})
                        continue
                else:
                    result = self.analyze_audio(x['subtitles'], settings=settings, outputs=outputs)
                    results.append({'video_info':x,'result':result})
                    continue
            transcription = self.transcribe_video(x['url'], x['video_id'])
            x['transcribed'] = True
            result = self.analyze_audio(transcription['subtitles'], settings=settings, outputs=outputs)
            x.update(transcription)
            results.append({'video_info':x,'result':result})
        if n==1:
            if save_as:
                with open(self.temp_dir.strip('\\')+f'/{save_as}_result.pkl', 'wb') as f:
                    pickle.dump(results[0], f)
            if 'error' in results[0]['result']:
                raise InformError(results[0]['result']['error'])
            return results[0]
        else:
            if save_as:
                with open(self.temp_dir.strip('\\')+f'/{save_as}_results.pkl', 'wb') as f:
                    pickle.dump(results, f)
            return results
    
    def analyze_audio_file(self, file_path,
                           settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}},
                           outputs=['final_levels']):
        print('Preparing to transcribing')
        transcription = self.transcribe_audio(file_path)
        print('Analysing audio')
        result = self.analyze_audio(transcription['subtitles'], settings=settings, outputs=outputs)
        return result

    def seconds2time(self, seconds):
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def subtitles2srt(self, subtitles):
        srt = ''
        for i, x in enumerate(subtitles):
            srt += f"{i+1}\n{self.seconds2time(x['start'])} --> {self.seconds2time(x['end'])}\n{x['text']}\n\n"
        return srt




class YoutubeLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)





def general_float2grade(num):
    rounded = int(num)
    if num>12:
        return "College and above"
    elif num>=4:
        return f"{rounded}th to {rounded+1}th grade"
    elif num>=3:
        return "3rd to 4th grade"
    elif num>=2:
        return "2nd to 3rd grade"
    elif num>=1:
        return "1st to 2nd grade"
    else:
        return "Below 1st grade"


def dale_chall_float2grade(num):
    rounded = int(num*2-5)
    if num<5:
        return "4th grade or below"
    elif num<9:
        return f"{rounded}th to {rounded+1}th grade"
    else:
        return "College and above"

def spache_float2grade(num):
    rounded = int(num)
    if num>=4:
        return "4th grade or above";
    elif num>=3:
        return "3rd to 4th grade"
    elif num>=2:
        return "2nd to 3rd grade"
    elif num>=1:
        return "1st to 2nd grade"
    else:
        return "Below 1st grade"

def float2hsk(num):
    if num>=7:
        return "Native"
    elif num>=6:
        return "HSK7-9"
    else:
        return f"HSK{int(num)+1}"
    
def cefr2float(cefr):
    floats = {"-∞":-1, 0:0, '0':0, 'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5, '+∞':6, 'NATIVE':6}
    return floats.get(cefr.upper())

def standardisePos(pos):
    poses = {'adjective':'ADJ',
     'adverb':'ADV',
     'v':'VERB',
     'n':'NOUN',
     'preposition':'ADP',
     'prep':'ADP',
     'pronoun':'PRON',
     'conjunction':'CONJ',
     'sconj':'CONJ'}
    # get only alpha
    pos = re.sub(r'[^a-z]','',pos.lower())
    return poses.get(pos,pos.upper())
    