from .get_results import get_adaptor

class AdoLevelAdaptor:
    def __init__(self,analyser, openai_api_key):
        self.analyser = analyser
        self.openai_api_key = openai_api_key
        
    def adapt(self, text, target_level, target_adjustment=0.5, even=False, by="paragraph", min_piece_length=200, n=1, auto_retry=False, return_result=True, model="cefr"):

        return get_adaptor()
    