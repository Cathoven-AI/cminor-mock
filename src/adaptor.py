from .get_results import get_adaptor

class AdoLevelAdaptor:
    def __init__(self,text):
        self.text = text
        
    def adapt(self,text, target_level, target_adjustment, n, by, even, auto_retry, min_piece_length, return_result, model):
        return get_adaptor()
    