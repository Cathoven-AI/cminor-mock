from .get_results import get_cefr, get_catile, get_readability, get_video_analyzer, get_transcribe_video, get_video_info

class AdoTextAnalyzer:
    def __init__(self, openai_api_key=None):
        """ Initialize the AdoTextAnalyzer class with the OpenAI API key. """
        self.openai_api_key = openai_api_key

    def simplify(self,text, target_level, target_adjustment, n, by_sentence, auto_retry,return_result,up):
        pass

    def analyze_cefr(self,text, propn_as_lowest, intj_as_lowest, keep_min, return_phrase_count, custom_dictionary, return_sentences, return_wordlists, return_vocabulary_stats, return_tense_count, return_tense_term_count,return_tense_stats, return_clause_count, return_clause_stats,return_final_levels,return_result, return_modified_final_levels,settings,outputs, v):
        return get_cefr()

    def analyze_catile(self,text,return_result,v):
        return get_catile()

    def analyze_readability(self,text,language,return_grades,return_result):
        return get_readability()

class AdoVideoAnalyzer:
    def __init__(self, analyser, open_ai_api_key, temp_dir):
        self.analyser = analyser
        self.open_ai_api_key = open_ai_api_key
        self.temp_dir = temp_dir
    
    def get_video_info(self,url):
        return get_video_info()

    def transcribe_video(self,url,video_id):
        return get_transcribe_video()

    def analyze_audio(self,text,speak_duration,propn_as_lowest, intj_as_lowest, keep_min):
        return {}

    def analyze_youtube_video(self,url, transcribe, auto_transcribe, duration_limit, settings, outputs):
        return get_video_analyzer()
    

