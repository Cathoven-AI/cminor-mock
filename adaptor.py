import numpy as np
import openai, warnings, time, os, sys, ast
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "files/model_files/gpt_tokenizer"))

class AdoLevelAdaptor(object):
    def __init__(self, text_analyser, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.before_result = None
        self.analyser = text_analyser
        self.result = None


    def adapt(self, text, target_level, target_adjustment=0.5, even=False, by="paragraph", n=1, auto_retry=False, return_result=False, model="gpt-3.5-turbo"):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key
        text = text.replace("\u00A0", " ").replace('\xa0',' ').strip()

        self.t0 = time.time()
        self.openai_time = 0

        self.before_result = None
        self.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by=by, auto_retry=auto_retry, model=model)
        print(f"Total time taken: OpenAI {self.openai_time} seconds, everything else {time.time()-self.t0-self.openai_time} seconds")

        if return_result:
            return self.result

    def divide_piece(self, piece):
        max_length = 6000
        pieces = []
        n_pieces = int(np.ceil(len(piece.split(' '))/max_length))
        if n_pieces<=1:
            return [piece]
        else:
            sents = sent_tokenize(piece)
            length = len(sents)//n_pieces
            for i in range(n_pieces-1):
                pieces.append(' '.join(sents[length*i:length*(i+1)]))
            pieces.append(' '.join(sents[length*(n_pieces-1):]))
            return pieces


    def start_adapt(self, text, target_level, target_adjustment=0.5, even=False, n=1, by="paragraph", auto_retry=False, model="gpt-3.5-turbo"):
        if by=='sentence':
            by = "paragraph"
        n = max(1,min(n,5))

        if self.before_result is None:
            before_result = self.analyser.analyze_cefr(text,return_sentences=True, return_wordlists=False,return_vocabulary_stats=False,
                            return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                            return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)
        else:
            before_result = self.before_result
        before_levels = before_result['final_levels']
        if len(before_result['modified_final_levels'])>1:
            modified_before_levels = before_result['modified_final_levels']
        else:
            modified_before_levels = None
        after_levels = None
        modified_after_levels = None
        adaptations = []

        pieces = []
        '''
        if by=='sentence':
            sentence_levels = []
            for _,v in result['sentences'].items():
                piece = ''
                for i in range(len(v['whitespace'])):
                    piece += v['word'][i]+' '*v['whitespace'][i]
                pieces.append(piece)
                sentence_levels.append({"general_level":max(v['CEFR_vocabulary'],v['CEFR_clause']),"vocabulary_level":v['CEFR_vocabulary'],"clause_level":v['CEFR_clause']})
        '''
        if by=='paragraph':
            for piece in text.split('\n'):
                pieces += self.divide_piece(piece)
        else:
            pieces = self.divide_piece(text)

        if len(pieces)>0:
            pieces_reversed = pieces[::-1]
            pieces = []
            for piece in pieces_reversed:
                if len(piece.strip().split(' '))<10 and len(pieces)>0:
                    pieces[-1] = piece+'\n'+pieces[-1]
                else:
                    pieces.append(piece)
            if len(pieces[0].strip().split(' '))<10:
                pieces[1] = pieces[0]+'\n'+pieces[1]
                pieces = pieces[1:]
            pieces = pieces[::-1]

        n_pieces = len(pieces)

        for k,piece in enumerate(pieces):
            if len(piece.strip())==0:
                adaptations.append(piece)
                continue
            
            #if by=='sentence':
            #    piece_levels = sentence_levels[k]
            if n_pieces>1:
                result = self.analyser.analyze_cefr(piece,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                piece_levels = result['final_levels']
            else:
                piece_levels = before_levels

            if int(piece_levels['vocabulary_level'])>target_level:
                change_vocabulary = -1
            elif int(piece_levels['vocabulary_level'])<target_level and even:
                change_vocabulary = 1
            else:
                change_vocabulary = 0

            if int(piece_levels['clause_level'])>target_level:
                change_clause = -1
            elif int(piece_levels['clause_level'])<target_level and even:
                change_clause = 1
            else:
                change_clause = 0

            if change_vocabulary==0 and change_clause==0:
                if int(piece_levels['general_level'])<int(target_level):
                    change_vocabulary = 1
                    change_clause = 1
                else:
                    adaptations.append(piece)
                    continue

            openai_t0 = time.time()
            n_self_try = 3
            while n_self_try>0:
                try:
                    '''
                    if by=='sentence' and (change_vocabulary<0 or change_clause<0):
                        candidates = [self.simplify_sentence(
                            piece, target_level=target_level, target_adjustment=target_adjustment, 
                            change_vocabulary=change_vocabulary, change_clause=change_clause, contexts=pieces[k-2:k])]
                    '''
                    candidates = self.get_adaptation(
                        piece, target_level=target_level, target_adjustment=target_adjustment, n=n, 
                        change_vocabulary=change_vocabulary, change_clause=change_clause, model=model)
                    break
                except Exception as e:
                    n_self_try -= 1
                    if n_self_try==0:
                        self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                        return
                    print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

            self.openai_time += time.time()-openai_t0

            if candidates is None:
                return

            min_difference = 100
            min_difference_std = -1
            adaptation = piece.strip()
            for candidate in candidates:
                if len(candidate.strip().split(' '))<len(piece.strip().split(' '))*0.3:
                    continue
                result = self.analyser.analyze_cefr(candidate,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)

                if int(result['final_levels']['general_level'])==target_level:
                    adaptation = candidate
                    after_levels = result['final_levels']
                    modified_after_levels = None
                    break
                elif n_pieces==1 and auto_retry==False and len(result['modified_final_levels'])>0 and int(result['modified_final_levels'][-1]['final_levels']['general_level'])==target_level:
                    adaptation = candidate
                    after_levels = result['final_levels']
                    modified_after_levels = result['modified_final_levels'][-1]
                    break

                if change_vocabulary:
                    vocabulary_difference = abs(result['final_levels']['vocabulary_level']-(target_level+target_adjustment))
                else:
                    vocabulary_difference = abs(result['final_levels']['vocabulary_level']-before_levels['vocabulary_level'])

                tense_difference = abs(result['final_levels']['tense_level']-(target_level+target_adjustment))

                if change_clause:
                    clause_difference = abs(result['final_levels']['clause_level']-(target_level+target_adjustment))
                else:
                    clause_difference = abs(result['final_levels']['clause_level']-before_levels['clause_level'])

                difference = vocabulary_difference+tense_difference+clause_difference
                difference_std = np.std([vocabulary_difference,tense_difference,clause_difference])
                if difference<1:
                    adaptation = candidate
                    after_levels = result['final_levels']
                    if len(result['modified_final_levels'])>0:
                        modified_after_levels = result['modified_final_levels'][-1]
                    break
                elif difference<min_difference or difference==min_difference and difference_std<min_difference_std:
                    adaptation = candidate
                    after_levels = result['final_levels']
                    if len(result['modified_final_levels'])>0:
                        modified_after_levels = result['modified_final_levels'][-1]
                    min_difference = difference
                    min_difference_std = difference_std
            adaptations.append(adaptation)

        if by=='paragraph':
            after_text = '\n'.join(adaptations)
        else:
            after_text = ' '.join(adaptations)
        
        if n_pieces>1:
            modified_after_levels = None
            after_result = self.analyser.analyze_cefr(after_text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                            return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                            return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)
            after_levels = after_result['final_levels']
            if len(after_result['modified_final_levels'])>0:
                modified_after_levels = after_result['modified_final_levels'][-1]

        if after_levels is None:
            after_levels = before_levels
            modified_after_levels = modified_before_levels

        if auto_retry and int(after_levels['general_level'])!=target_level and (modified_after_levels is None or int(modified_after_levels['final_levels']['general_level'])!=target_level):
            return self.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by='piece', auto_retry=False, model=model)

        self.result = {'adaptation':after_text, 'before':before_levels, 'after': after_levels, 'modified_after_levels': modified_after_levels}


    def get_adaptation(self, text, target_level, target_adjustment=0.5, n=1, change_vocabulary=-1, change_clause=-1, model="gpt-3.5-turbo"):
        prompt = self.construct_prompt(text=text, target_level=target_level, target_adjustment=target_adjustment, change_vocabulary=change_vocabulary, change_clause=change_clause)
        if prompt=="":
            return []

        n_tokens = len(gpt_tokenizer.encode(prompt)) + len(gpt_tokenizer.encode(text))*(n+1)

        if n_tokens>6000:
            model="gpt-4"

        if model=="gpt-4":
            if n_tokens>8000:
                model_name = "gpt-4-32k"
            else:
                model_name = "gpt-4"
        else:
            if n_tokens>4000:
                model_name = "gpt-3.5-turbo-16k"
            else:
                model_name = "gpt-3.5-turbo"

        prompt = prompt+f"\nPassage:\n```{text}```"

        completion = openai.ChatCompletion.create(
            model=model_name, n=n,
            messages=[{"role": "user", "content": prompt}]
        )

        adaptations = []
        if False:#change_vocabulary<0:
            for x in completion['choices']:
                x = x['message']['content'].strip()
                try:
                    adaptations.append(ast.literal_eval(x)['text'])
                except:
                    pass
        else:
            for x in completion['choices']:
                x = x['message']['content'].strip()
                if x.lower().startswith("simplified: "):
                    x = x[12:].strip()
                elif x.lower().startswith("simplified version: "):
                    x = x[20:].strip()
                adaptations.append(x)
        return adaptations
    
    def construct_prompt(self, text, target_level, target_adjustment=0.5, change_vocabulary=-1, change_clause=-1):
        prompt = ''
        int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
        level = int2cefr[target_level]
        max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
        
        if False:#change_vocabulary<0:
            prompt = f'''
You are an English textbook editor. Your task is to write texts at CEFR {level} level.

Follow these steps when writing:
1. Use summarize all the ideas of a text as bullet points
2. To recreate the text at CEFR {level} level, use one or more simple sentences to rewrite each point.

A text at CEFR {level} level should meet the following requirements:
1. The vocabulary should be simple and below CEFR {level} level. Don't use technical or academic words. Paraphrase technical or academic words. For example:
    transport => move
    shrink => get smaller
    pregnancy => having a baby
    have serious consequences => bad things will happen
    anaesthesia => using drugs to make people feel no pain
2. Each sentence is not longer than {max_length} words.
3. There should be no complex grammartical clauses (like noun clauses, relative clauses, etc.). If there is any, break it down into several short sentences.  For example:
```"Studying galaxies helps us understand more about how the universe has changed over time." => "The universe has changed over time. Studying galaxies helps us understand more about this."```


    
Original text:
```
{text}
```
'''
            prompt += '''
Return both the bullet points and the new text in a Python dictionary like this:
```{'bullet_points':[point 1, point 2, ...], 'text': the new text}```.
    '''
        else:
            levels = [int2cefr[i] for i in range(target_level+1)]
            levels = ', '.join(levels[:-1]) + f' and {levels[-1]}'

            examples = '''\nExamples of replacing difficult words and long clauses:
transport => move
shrink => get smaller
pregnancy => having a baby
have serious consequences => bad things will happen
anaesthesia => using drugs to make people feel no pain
Studying galaxies helps us understand more about how the universe has changed over time. => The universe has changed over time. Studying galaxies helps us understand more about this.'''

            if change_clause<0:
                max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment-1+1.5)/np.log(1.1),0))
                if change_vocabulary<0:
                    prompt = f'''Your task is to rewrite this passage so that a child can easily understand.
For each sentence, follow these rules:
1. Keep the details. Do not just summerize.
2. Replace difficult words and use mainly words at CEFR {levels} levels.
3. Use no more than {max_length} words.
4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
''' + examples
                elif change_vocabulary>0:
                    prompt = f'''Your task is to replace easy words in this passage so that most of the words are at CEFR {int2cefr[target_level]} level.
For each sentence, follow these rules:
1. Keep the details. Do not just summerize.
2. Use mainly words at CEFR {int2cefr[target_level]} level.
3. Use no more than {max_length} words.
4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
'''
                else:
                    prompt = f'''Your task is to rewrite this passage so that the sentence structure is simplier.
For each sentence, follow these rules:
1. Keep the details. Do not just summerize.
2. Use only the vocabulary in the original passage.
3. Use no more than {max_length} words.
4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
'''
            elif change_clause>0:
                max_length = int(round(np.log(target_level+1+target_adjustment+1.5)/np.log(1.1),0))
                min_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
                if change_vocabulary<0:
                    prompt = f'''Your task is to rewrite this passage to make the sentence structure more complex.
For each sentence, follow these rules:
1. Do not make the vocabulary more complex.
2. Replace difficult words and use only words at CEFR {levels} levels so that a child can easily understand.
3. Use {min_length} to {max_length} words.
''' + examples
                elif change_vocabulary>0:
                    prompt = f'''Your task is to rewrite this passage to make it more complex.
For each sentence, follow these rules:
1. Use mainly words at CEFR {int2cefr[target_level]} level.
2. Use {min_length} to {max_length} words.
'''
                else:
                    prompt = f'''Your task is to rewrite this passage to make the sentence structure more complex.
For each sentence, follow these rules:
1. Do not make the vocabulary more complex.
2. Use only the vocabulary in the original passage.
3. Use {min_length} to {max_length} words.
'''
            else:
                if change_vocabulary<0:
                    prompt = f'''Your task is to replace difficult words with easier ones in this passage.
For each sentence, follow these rules:
1. Try not to change the sentence structure.
2. Replace difficult words and use only words at CEFR {levels} levels so that a child can easily understand.
'''+examples
                elif change_vocabulary>0:
                    prompt = f'''Your task is to replace easier words with more difficult ones in this passage.
For each sentence, follow these rules:
1. Try not to change the sentence structure.
2. Use only words at CEFR {int2cefr[target_level]} level.
'''
                else:
                    return ''
            prompt = prompt+f"\nPassage:\n```{text}```"
        return prompt