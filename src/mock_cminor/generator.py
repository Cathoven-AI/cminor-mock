from .get_results import get_question_generator, get_text_generator, get_writing_assessment_revise, get_writing_assessment_enhance

class AdoTextGenerator:
    def __init__(self, analyser, openai_api_key=None, antropic_api_key=None):
        self.analyser = analyser
        self.openai_api_key = openai_api_key
        self.antropic_api_key = antropic_api_key

    def create_text(self,
                      level,n_words=300,topic=None,grammar=None, genre=None, 
                      settings={'ignore_keywords':True,'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'custom_dictionary':{}},
                      outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels','modified_final_levels'], 
                      keywords=None
                      ):
        
        return get_text_generator()

class AdoWritingAssessor:
    def __init__(self, analyser, openai_api_key, antropic_api_key=None):
        self.analyser = analyser
        self.openai_api_key = openai_api_key
        self.antropic_api_key = antropic_api_key

    def revise(self,text, comment=False, writing_language="English", comment_language=None, auto_retry=2):
        return get_writing_assessment_revise()

    def enhance(self,text, level=None, comment=False, writing_language="English", comment_language=None, auto_retry=2):
        return get_writing_assessment_enhance()

class AdoQuestionGenerator:
    def __init__(self, text_analyser, video_analyser, openai_api_key=None, antropic_api_key=None):
        self.text_analyser = text_analyser
        self.video_analyser = video_analyser
        self.openai_api_key = openai_api_key
        self.antropic_api_key = antropic_api_key

        

    def generate_questions(self, text=None, url=None, words=None, n=10, kind='multiple_choice', skill='reading', level=None, 
                           answer_position=False, explanation=False, question_language=None, explanation_language=None, auto_retry=3,
                           transcribe=False, duration_limit=900):
        
        return get_question_generator()
    
