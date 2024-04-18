from .get_results import get_adaptor

class AdoLevelAdaptor:
    def __init__(self,analyser, openai_api_key):
        self.analyser = analyser
        self.openai_api_key = openai_api_key
        
    def adapt(self,text, target_level, target_adjustment, n, by, even, auto_retry, min_piece_length, return_result, model):
        return get_adaptor()
    