from .get_results import get_cefr, get_catile, get_readability, get_video_analyzer, get_transcribe_video, get_video_info

class AdoTextAnalyzer:
    def __init__(self, openai_api_key=None):
        """ Initialize the AdoTextAnalyzer class with the OpenAI API key. """
        self.openai_api_key = openai_api_key

    def simplify(self,text, target_level, target_adjustment, n, by_sentence, auto_retry,return_result,up):
        pass

    def analyze_cefr(self,
                     text,
                     settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}},
                     outputs=['final_levels'],
                     propn_as_lowest=True,intj_as_lowest=True,keep_min=True,custom_dictionary={},
                     return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                     return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                     return_clause_stats=True,return_phrase_count=True,return_final_levels=True,
                     return_result=True,return_modified_final_levels=False, v=1):

        return get_cefr()

    def analyze_catile(self,text,return_result=True,v=1):
        return get_catile()

    def analyze_readability(self,text,language="en",return_grades=False,return_result=True):
        return get_readability()

class AdoVideoAnalyzer:
    def __init__(self, analyser, openai_api_key, temp_dir):
        self.analyser = analyser
        self.openai_api_key = openai_api_key
        self.temp_dir = temp_dir
    
    def get_video_info(self,url):
        return get_video_info()

    def transcribe_video(self,url,video_id):
        return get_transcribe_video()

    def analyze_audio(self,text,speak_duration,propn_as_lowest, intj_as_lowest, keep_min):
        return {}

    def analyze_youtube_video(self,url, transcribe=False, auto_transcribe=True, verbose=False, save_as=None, duration_limit=900, settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'as_wordlist':False,'custom_dictionary':{}}, outputs=['final_levels']):
        return get_video_analyzer()
    

