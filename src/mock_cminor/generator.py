from .get_results import get_question_generator, get_text_generator, get_writing_assessment_revise, get_writing_assessment_enhance

class AdoTextGenerator:
    def __init__(self, analyser, openai_api_key):
        self.analyser = analyser
        self.openai_api_key = openai_api_key

    def create_text(self,level,n_words,topic,grammar, genre, settings, outputs ,keywords):
        return get_text_generator()

class AdoWritingAssessor:
    def __init__(self, analyser, openai_api_key):
        self.analyser = analyser
        self.openai_api_key = openai_api_key

    def revise(self,text, comment, writing_language, comment_language):
        return get_writing_assessment_revise()

    def enhance(self,text, level, comment, writing_language, comment_language):
        return get_writing_assessment_enhance()

class AdoQuestionGenerator:
    def __init__(self, text_analyser, video_analyser, openai_api_key):
        self.text_analyser = text_analyser
        self.video_analyser = video_analyser
        self.openai_api_key = openai_api_key

    def generate_questions(self,text,n,kind,auto_retry, explanation, explanation_language,answer_position,question_language, words, skill, level, url, transcribe, duration_limit):
        return get_question_generator()
    