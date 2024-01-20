import numpy as np
import openai, warnings, time, os, sys, ast
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from .utils import InformError, check_level_input_and_int

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "files/model_files/gpt_tokenizer"))

class AdoLevelAdaptor(object):
    def __init__(self, text_analyser, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.before_result = None
        self.analyser = text_analyser
        self.result = None


    def adapt(self, text, target_level, target_adjustment=0.5, even=False, by="paragraph", min_piece_length=200, n=1, auto_retry=False, return_result=True, model="gpt-3.5-turbo"):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key

        target_level = check_level_input_and_int(target_level)
        if self.analyser.detect(text.replace('\n',' '))['lang'] != 'en':
            raise InformError("Language not supported. Please use English.")

        text = text.replace("\u00A0", " ").replace('\xa0',' ').strip()

        self.t0 = time.time()
        self.openai_time = 0

        self.before_result = None
        self.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by=by, min_piece_length=min_piece_length, auto_retry=auto_retry, model=model)
        print(f"Total time taken: OpenAI {self.openai_time} seconds, everything else {time.time()-self.t0-self.openai_time} seconds")

        if return_result:
            return self.result

    def divide_piece(self, piece, min_piece_length=2000, by='piece'):
        min_piece_length = min(min_piece_length,2000)
        pieces = []
        n_pieces = int(np.ceil(len(piece.split(' '))/min_piece_length))
        if n_pieces<=1:
            return [piece]
        else:
            if by=='paragraph':
                sents = []
                for x in piece.split('\n'):
                    x = x.strip(' \n')
                    if x!='':
                        sents.append(x+'\n')
            else:
                sents = sent_tokenize(piece)
            length = len(sents)//n_pieces
            for i in range(n_pieces-1):
                pieces.append(' '.join(sents[length*i:length*(i+1)]))
            pieces.append(' '.join(sents[length*(n_pieces-1):]))

            if len(pieces)>0:
                pieces_reversed = pieces[::-1]
                pieces = []
                for piece in pieces_reversed:
                    if len(piece.strip().split(' '))<20 and len(pieces)>0:
                        pieces[-1] = piece+'\n'+pieces[-1]
                    else:
                        pieces.append(piece)
                if len(pieces[0].strip().split(' '))<20:
                    pieces[1] = pieces[0]+'\n'+pieces[1]
                    pieces = pieces[1:]
                pieces = pieces[::-1]
            return pieces


    def start_adapt(self, text, target_level, target_adjustment=0.5, even=False, n=1, by="paragraph", min_piece_length=100, auto_retry=False, model="gpt-3.5-turbo"):
        if by=='sentence':
            by = "paragraph"
        auto_retry = int(auto_retry)

        n = max(1,min(n,5))

        if self.before_result is None:
            before_result = self.analyser.analyze_cefr(text,return_sentences=True, return_wordlists=True,return_vocabulary_stats=False,
                            return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                            return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)
        else:
            before_result = self.before_result
        before_levels = before_result['final_levels']
        if int(before_levels['general_level'])<target_level and target_level>=3:
            n = min(n,3)
        if len(before_result['modified_final_levels'])>1:
            modified_before_levels = before_result['modified_final_levels']
        else:
            modified_before_levels = None
        after_result = None
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
            pieces += self.divide_piece(text, min_piece_length=min_piece_length, by=by)
        else:
            pieces = self.divide_piece(text, min_piece_length=2000, by=by)



        n_pieces = len(pieces)

        for k,piece in enumerate(pieces):
            if len(piece.strip())==0:
                adaptations.append(piece)
                continue
            
            #if by=='sentence':
            #    piece_levels = sentence_levels[k]
            if n_pieces>1:
                piece_result = self.analyser.analyze_cefr(piece,return_sentences=True, return_wordlists=True,return_vocabulary_stats=False,
                                return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                                return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False)
                piece_levels = piece_result['final_levels']
            else:
                piece_levels = before_levels
                piece_result = before_result

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
            adaptation = piece
            if change_vocabulary>0 or change_clause>0:
                adaptation, after_result = self.get_adaptation(adaptation, target_level, target_adjustment=target_adjustment, n=n, direction="up", model=model)
                if int(after_result['final_levels']['general_level'])>target_level:
                    if int(after_result['final_levels']['vocabulary_level'])>target_level:
                        adaptation, after_result = self.get_adaptation(self.tag_difficult_words(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='vocabulary',direction="down", model=model)
                    if int(after_result['final_levels']['clause_level'])>target_level:
                        adaptation, after_result = self.get_adaptation(self.tag_difficult_sentences(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='clause',direction="down", model=model)
            else:
                after_result = piece_result
                if change_vocabulary<0:
                    adaptation, after_result = self.get_adaptation(self.tag_difficult_words(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='vocabulary',direction="down",model=model)
                if change_clause<0 and int(after_result['final_levels']['clause_level'])>target_level:
                    adaptation, after_result = self.get_adaptation(self.tag_difficult_sentences(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='clause',direction="down", model=model)
            adaptations.append(adaptation.strip('\n'))

        if by=='paragraph':
            after_text = '\n'.join(adaptations)
        else:
            after_text = ' '.join(adaptations)
        
        if len(adaptations)>1:
            after_result = self.analyser.analyze_cefr(after_text,return_sentences=False, return_wordlists=False,return_vocabulary_stats=False,
                            return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                            return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)
        if after_result is None:
            after_levels = before_levels
            modified_after_levels = modified_before_levels
        else:
            modified_after_levels = None
            after_levels = after_result['final_levels']
            if len(after_result['modified_final_levels'])>0:
                modified_after_levels = after_result['modified_final_levels'][-1]


        if auto_retry>0 and int(after_levels['general_level'])!=target_level and (modified_after_levels is None or int(modified_after_levels['final_levels']['general_level'])!=target_level):
            return self.start_adapt(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by='piece', auto_retry=auto_retry-1, model=model)

        self.result = {
            'adaptation':after_text,
            'before':before_levels,
            'after': after_levels,
            'modified_after_levels': modified_after_levels, 
            'before_str':{k:float2cefr(v) for k,v in before_levels.items()},
            'after_str':{k:float2cefr(v) for k,v in after_levels.items()},
            'before_exam_stats':self.analyser.cefr.float2exams(before_levels['general_level']),
            'after_exam_stats':self.analyser.cefr.float2exams(after_levels['general_level']),
        }


    def tag_difficult_words(self, result, target_level):
        sentences = result['sentences']
        wordlists = result['wordlists']
        
        threshold = target_level+1
        p = 0.5
        done = False
        total_words = {}
        for i in range(-1,7):
            if i in wordlists:
                total_words[i] = sum(wordlists[i]['size'])

        for i in reversed(range(-1,7)):
            if done:
                break
            if i not in total_words:
                continue
            n = total_words[i]
            m = 0
            while m<n:
                levels = []
                for k,v in total_words.items():
                    if k<i:
                        levels += [k]*v
                    elif k>i:
                        levels += [-1]*v
                    else:
                        levels += [-1]*m+[k]*(n-m)
                if self.analyser.cefr.estimate_95(levels)<target_level+0.8:
                    threshold = i
                    p = m/n
                    done = True
                    break
                m += 1

        not_tagged = True
        tagged_text = ''
        for s in sentences.values():
            for i in range(len(s['lemma'])):
                if s['CEFR'][i]>threshold:
                    tagged_text += "<b>"+s["word"][i]+"</b>"+' '*s["whitespace"][i]
                    not_tagged = False
                elif s['CEFR'][i]==threshold and (not_tagged or np.random.rand()<p):
                    tagged_text += "<b>"+s["word"][i]+"</b>"+' '*s["whitespace"][i]
                    not_tagged = False
                else:
                    tagged_text += s["word"][i]+' '*s["whitespace"][i]
        return tagged_text
    
    def tag_difficult_sentences(self, result, target_level):
        not_tagged = True
        tagged_text = ''
        for s in result['sentences'].values():
            sent = ''
            for i in range(len(s['lemma'])):
                sent += s["word"][i]+' '*s["whitespace"][i]
            if int(s['CEFR_clause'])>target_level+1:
                tagged_text += "<i>"+sent+"</i>"
                not_tagged = False
            elif int(s['CEFR_clause'])==target_level+1 and (not_tagged or np.random.rand()<0.5):
                tagged_text += "<i>"+sent+"</i>"
                not_tagged = False
            else:
                tagged_text += sent
        return tagged_text

    def get_best_candidate(self, candidates, target_level, target_adjustment=0.5, on='general_level'):
        min_difference = 100
        finalist_result = None
        for candidate in candidates:
            result = self.analyser.analyze_cefr(candidate,return_sentences=True, return_wordlists=True, return_vocabulary_stats=False,
                            return_tense_count=False,return_tense_term_count=False,return_tense_stats=False,return_clause_count=False,
                            return_clause_stats=False,return_phrase_count=False,return_final_levels=True,return_result=True,clear_simplifier=False,return_modified_final_levels=True)
            if int(result['final_levels']['vocabulary_level'])==target_level and int(result['final_levels']['vocabulary_level']+0.2)==target_level and int(result['final_levels']['clause_level'])==target_level and int(result['final_levels']['clause_level']+0.2)==target_level:
                finalist = candidate
                finalist_result = result
                break
            else:
                if on=='general_level':
                    diff = abs(result['final_levels']['vocabulary_level']-(target_level+target_adjustment))+abs(result['final_levels']['clause_level']-(target_level+target_adjustment))
                elif on!='vocabulary_level':
                    diff = abs(int(result['final_levels']['vocabulary_level'])-target_level)*10+abs(result['final_levels']['clause_level']-(target_level+target_adjustment))
                else:
                    diff = abs(int(result['final_levels']['clause_level'])-target_level)+abs(result['final_levels']['vocabulary_level']-(target_level+target_adjustment))
                if diff<min_difference:
                    finalist = candidate
                    finalist_result = result
                    min_difference = diff
        return finalist, finalist_result

    def get_adaptation(self, text, target_level, target_adjustment=0.5, n=1, change='vocabulary', direction="down",model="gpt-3.5-turbo"):
        n = min(n,10)
        model_name = 'gpt-4-1106-preview'
        n_per_call = min(max(1,int(8000/len(text.split(' '))-1)),n)
        n_self_try = 3
        candidates = []

        prompt = self.construct_prompt(text=text, target_level=target_level, target_adjustment=target_adjustment, change=change, direction=direction)
        while n_self_try>0 and len(candidates)<n:
            try:
                openai_t0 = time.time()
                completion = openai.ChatCompletion.create(
                    model=model_name, n=max(n_per_call,n-len(candidates)),
                    messages=[{"role": "user", "content": prompt}]
                )
                self.openai_time += time.time()-openai_t0
            except Exception as e:
                n_self_try -= 1
                if n_self_try==0:
                    self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                    return
                print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

            for x in completion['choices']:
                x = x['message']['content'].strip()
                x = x.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','')
                candidates.append(x)

        if direction=="down":
            if change!='vocabulary':
                on = 'clause_level'
            else:
                on = 'vocabulary_level'
        else:
            on = 'general_level'
        finalist, finalist_result = self.get_best_candidate(candidates, target_level, target_adjustment=target_adjustment,on=on)
        return finalist, finalist_result

    
    def construct_prompt(self, text, target_level, target_adjustment=0.5, change='vocabulary',direction='down'):
        int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
        level = int2cefr[target_level]
        prompt = f'''You are an English textbook editor. Your task is to modify a text to fit CEFR {level} level.
        
Original text:
```
{text}
```
'''
        if direction == 'down':
            if change!='vocabulary':
                max_length = int(round(np.log(target_level+1+target_adjustment+1.5)/np.log(1.1),0))
                min_length = max(1,int(round(np.log(target_level-1+target_adjustment+1.5)/np.log(1.1),0)))
                prompt += f'''\nIn this text, sentences within <i> tags are difficult for CEFR {level}.
Change the structure of these tagged sentences following these rules:
1. Use no more than {max_length} words for each sentence.
2. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
Examples of breaking long clauses:
"Studying galaxies helps us understand more about how the universe has changed over time." => "The universe has changed over time. Studying galaxies helps us understand more about this."

Do not change any other sentences not tagged.
Do not to replace any words in the original text.
'''
            else:
                levels = [int2cefr[i] for i in range(target_level+1)]
                levels = ', '.join(levels[:-1]) + f' and {levels[-1]}'
                prompt+=f'''\nIn this text, words within <b> tags are difficult for CEFR {level}. If these words are not necessary topic words, replace them with words at CEFR {levels} levels. Don't change any other words not tagged.

Examples of replacing difficult words:
    transport => move
    shrink => get smaller
    pregnancy => having a baby
    have serious consequences => bad things will happen
    anaesthesia => using drugs to make people feel no pain
                
Keep the details. Do not just summerize.
Do not change the original sentences' structure.
'''

        else:
            max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
            min_length = int(round(np.log(target_level-1+target_adjustment+1.5)/np.log(1.1),0))
            prompt += f'''\nThis text is too easy for CEFR {level} level. Rewrite it so that it satisfies these requirements:
1. There are many words at CEFR {level} level, but no words above CEFR {level} level.
2. Each sentence should have approximately {min_length} to {max_length} words.

'''
        prompt += '''
Do not remove titles and subtitles if there are any.

Output only the new text without any comments or tags.'''
        return prompt



#     def construct_prompt(self, text, target_level, target_adjustment=0.5, change_vocabulary=-1, change_clause=-1):
#         prompt = ''
#         int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
#         level = int2cefr[target_level]
#         max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
        
#         if False:#change_vocabulary<0:
#             prompt = f'''
# You are an English textbook editor. Your task is to write texts at CEFR {level} level.

# Follow these steps when writing:
# 1. Use summarize all the ideas of a text as bullet points
# 2. To recreate the text at CEFR {level} level, use one or more simple sentences to rewrite each point.

# A text at CEFR {level} level should meet the following requirements:
# 1. The vocabulary should be simple and below CEFR {level} level. Don't use technical or academic words. Paraphrase technical or academic words. For example:
#     transport => move
#     shrink => get smaller
#     pregnancy => having a baby
#     have serious consequences => bad things will happen
#     anaesthesia => using drugs to make people feel no pain
# 2. Each sentence is not longer than {max_length} words.
# 3. There should be no complex grammartical clauses (like noun clauses, relative clauses, etc.). If there is any, break it down into several short sentences.  For example:
# ```"Studying galaxies helps us understand more about how the universe has changed over time." => "The universe has changed over time. Studying galaxies helps us understand more about this."```


    
# Original text:
# ```
# {text}
# ```
# '''
#             prompt += '''
# Return both the bullet points and the new text in a Python dictionary like this:
# ```{'bullet_points':[point 1, point 2, ...], 'text': the new text}```.
#     '''
#         else:
#             levels = [int2cefr[i] for i in range(target_level+1)]
#             levels = ', '.join(levels[:-1]) + f' and {levels[-1]}'

#             examples = '''\nExamples of replacing difficult words and long clauses:
# transport => move
# shrink => get smaller
# pregnancy => having a baby
# have serious consequences => bad things will happen
# anaesthesia => using drugs to make people feel no pain
# Studying galaxies helps us understand more about how the universe has changed over time. => The universe has changed over time. Studying galaxies helps us understand more about this.'''

#             if change_clause<0:
#                 max_length = int(round(np.log(target_level+target_adjustment+1.5)/np.log(1.1),0))
#                 min_length = int(round(np.log(target_level+target_adjustment-1+1.5)/np.log(1.1),0))
#                 if change_vocabulary<0:
#                     prompt = f'''Your task is to rewrite this passage so that a child can easily understand.
# For each sentence, follow these rules:
# 1. Keep the details. Do not just summerize.
# 2. Replace difficult words and use mainly words at CEFR {levels} levels.
# 3. Use no more than {max_length} words.
# 4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
# 5. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# ''' + examples
#                 elif change_vocabulary>0:
#                     prompt = f'''Your task is to replace easy words in this passage so that most of the words are at CEFR {int2cefr[target_level]} level.
# For each sentence, follow these rules:
# 1. Keep the details. Do not just summerize.
# 2. Use mainly words at CEFR {int2cefr[target_level]} level.
# 3. Use no more than {max_length} words.
# 4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
# 5. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''
#                 else:
#                     prompt = f'''Your task is to rewrite this passage so that the sentence structure is simplier.
# For each sentence, follow these rules:
# 1. Keep the details. Do not just summerize.
# 2. Use only the vocabulary in the original passage.
# 3. Use no more than {max_length} words.
# 4. If a sentence has more than {max_length} words, break it down by seperating the subordinate clauses as new sentences.
# 5. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''
#             elif change_clause>0:
#                 max_length = int(round(np.log(target_level+1+target_adjustment+1.5)/np.log(1.1),0))
#                 min_length = int(round(np.log(target_level-0.5+target_adjustment+1.5)/np.log(1.1),0))
#                 if change_vocabulary<0:
#                     prompt = f'''Your task is to rewrite this passage to make the sentence structure more complex.
# For each sentence, follow these rules:
# 1. Do not make the vocabulary more complex.
# 2. Replace difficult words and use only words at CEFR {levels[:4]} levels so that a child can easily understand.
# 3. Use {min_length} to {max_length} words.
# 4. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# ''' + examples
#                 elif change_vocabulary>0:
#                     prompt = f'''Your task is to rewrite this passage to make it more complex.
# For each sentence, follow these rules:
# 1. Use mainly words at CEFR {int2cefr[target_level]} level.
# 2. Use {min_length} to {max_length} words.
# 3. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''
#                 else:
#                     prompt = f'''Your task is to rewrite this passage to make the sentence structure more complex.
# For each sentence, follow these rules:
# 1. Do not make the vocabulary more complex.
# 2. Use only the vocabulary in the original passage.
# 3. Use {min_length} to {max_length} words.
# 4. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''
#             else:
#                 if change_vocabulary<0:
#                     prompt = f'''Your task is to replace difficult words with easier ones in this passage.
# For each sentence, follow these rules:
# 1. Try not to change the sentence structure.
# 2. Replace difficult words and use only words at CEFR {levels} levels so that a child can easily understand.
# 3. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''+examples
#                 elif change_vocabulary>0:
#                     prompt = f'''Your task is to replace easier words with more difficult ones in this passage.
# For each sentence, follow these rules:
# 1. Try not to change the sentence structure.
# 2. Use only words at CEFR {int2cefr[target_level]} level.
# 3. Don't add notes not related to the content, like "(number of words)", "[the number of sentence]".
# '''
#                 else:
#                     return ''
#             prompt = prompt+f"\nPassage:\n```{text}```"
#         return prompt


def float2cefr(num):
    cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
    output = cefr.get(int(num),"Native")
    if num<6:
        s = "%.1f" % num
        output += s[s.index('.'):]
    return output