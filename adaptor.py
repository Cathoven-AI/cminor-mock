import numpy as np
import pandas as pd
import warnings, time, os, sys, ast, asyncio, nest_asyncio, httpx
from openai import AsyncOpenAI, OpenAI
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from .utils import InformError, clean_target_level_input

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

nest_asyncio.apply()
gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "files/model_files/gpt_tokenizer"))
df_catile_examples = pd.read_csv(os.path.join(BASE_DIR, 'files/model_files/catile_example_texts.csv'))

class AdoLevelAdaptor(object):
    def __init__(self, text_analyser, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.before_result = None
        self.analyser = text_analyser
        self.result = None
        self.async_client = None
        self.client = None


    def adapt(self, text, target_level, target_adjustment=0.5, even=False, by="paragraph", min_piece_length=200, n=1, auto_retry=False, return_result=True, model="cefr"):
        self.t0 = time.time()
        self.openai_time = 0

        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            self.async_client = AsyncOpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(120, connect=5))
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(120, connect=5))

        # if self.analyser.detect(text.replace('\n',' '))['lang'] != 'en':
        #     raise InformError("Language not supported. Please use English.")
        text = text.replace("\u00A0", " ").replace('\xa0',' ').strip()
        auto_retry = max(0,min(int(auto_retry),5))
        self.before_result = None
        self.example_texts = None

        model = model.lower()
        target_level = clean_target_level_input(target_level, model=model)
        if model == 'catile':
            min_piece_length = max(min_piece_length, 2000)
            if target_level<500:
                n = max(1,min(n,5))
                self.start_adapt_catile(text, target_level, n=1, min_piece_length=min_piece_length, auto_retry=max(n-1,auto_retry))
            else:
                n = max(1,min(n,int(3-min_piece_length/200)))
                self.start_adapt_catile(text, target_level, n=n, min_piece_length=min_piece_length, auto_retry=auto_retry)
        else:
            n = max(1,min(n,int(6-min_piece_length/200)))
            self.start_adapt_cefr(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by=by, min_piece_length=min_piece_length, auto_retry=auto_retry)

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


    def start_adapt_cefr(self, text, target_level, target_adjustment=0.5, even=False, n=1, by="piece", min_piece_length=200, auto_retry=False):

        if self.before_result is None:
            before_result = self.analyser.analyze_cefr(text,outputs=['sentences','wordlists','final_levels','modified_final_levels'], v=2)
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
        pieces = self.divide_piece(text, min_piece_length=min_piece_length, by=by)
        pieces = [x.strip() for x in pieces if len(x.strip())>0]

        n_pieces = len(pieces)
        adaptations = [None]*n_pieces
        piece_infos_dict = {}
        for k,piece in enumerate(pieces):
            #if by=='sentence':
            #    piece_levels = sentence_levels[k]
            if n_pieces>1:
                piece_result = self.analyser.analyze_cefr(piece, outputs=['sentences','wordlists','final_levels'], v=2)
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
                    adaptations[k] = piece
                    continue
            # adaptation = piece
            # if change_vocabulary>0 or change_clause>0:
            #     adaptation, after_result = self.get_adaptation(adaptation, target_level, target_adjustment=target_adjustment, n=n, direction="up")
            #     if int(after_result['final_levels']['general_level'])>target_level:
            #         if int(after_result['final_levels']['vocabulary_level'])>target_level:
            #             adaptation, after_result = self.get_adaptation(self.tag_difficult_words(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='vocabulary',direction="down")
            #         if int(after_result['final_levels']['clause_level'])>target_level:
            #             adaptation, after_result = self.get_adaptation(self.tag_difficult_sentences(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='clause',direction="down")
            # else:
            #     after_result = piece_result
            #     if change_vocabulary<0:
            #         adaptation, after_result = self.get_adaptation(self.tag_difficult_words(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='vocabulary',direction="down")
            #     if change_clause<0 and int(after_result['final_levels']['clause_level'])>target_level:
            #         adaptation, after_result = self.get_adaptation(self.tag_difficult_sentences(after_result,target_level), target_level, target_adjustment=target_adjustment, n=n, change='clause',direction="down")
            # adaptations.append(adaptation.strip('\n'))
                
            if change_vocabulary>0 or change_clause>0:
                piece_infos_dict[k] = {'text':piece,'target_level':target_level,'target_adjustment':target_adjustment,'change':'vocabulary','direction':'up','model':'cefr', 'stage':1}
            else:
                if change_vocabulary<0:
                    piece_infos_dict[k] = {'text':self.tag_difficult_words(piece_result,target_level),'target_level':target_level,'target_adjustment':target_adjustment,'change':'vocabulary','direction':'down','model':'cefr', 'stage':2}
                elif change_clause<0 and int(piece_result['final_levels']['clause_level'])>target_level:
                    piece_infos_dict[k] = {'text':self.tag_difficult_words(piece_result,target_level),'target_level':target_level,'target_adjustment':target_adjustment,'change':'clause','direction':'down','model':'cefr', 'stage':3}
        
        while len(piece_infos_dict)>0:
            candidates_infos_dict = self.gather_candidates(piece_infos_dict)
            finalists_dict = self.get_finalists(candidates_infos_dict)
            for k,v in finalists_dict.items():
                adaptation, after_result = v
                if piece_infos_dict[k]['stage']==1:
                    if int(after_result['final_levels']['general_level'])>target_level:
                        if int(after_result['final_levels']['vocabulary_level'])>target_level:
                            piece_infos_dict[k] = {'text':self.tag_difficult_words(after_result,target_level),'target_level':target_level,'target_adjustment':target_adjustment,'change':'vocabulary','direction':'down','model':'cefr', 'stage':2}
                            continue
                        elif int(after_result['final_levels']['clause_level'])>target_level:
                            piece_infos_dict[k] = {'text':self.tag_difficult_sentences(after_result,target_level),'target_level':target_level,'target_adjustment':target_adjustment,'change':'clause','direction':'down','model':'cefr', 'stage':3}
                            continue
                elif piece_infos_dict[k]['stage']==2 and int(after_result['final_levels']['clause_level'])>target_level:
                    piece_infos_dict[k] = {'text':self.tag_difficult_sentences(after_result,target_level),'target_level':target_level,'target_adjustment':target_adjustment,'change':'clause','direction':'down','model':'cefr', 'stage':3}
                    continue
                adaptations[k] = adaptation.strip('\n')
                del piece_infos_dict[k]


        if by=='paragraph':
            after_text = '\n'.join(adaptations)
        else:
            after_text = ' '.join(adaptations)
        
        if len(adaptations)>1:
            after_result = self.analyser.analyze_cefr(after_text,outputs=['final_levels','modified_final_levels'], v=2)
        if after_result is None:
            after_levels = before_levels
            modified_after_levels = modified_before_levels
        else:
            modified_after_levels = None
            after_levels = after_result['final_levels']
            if len(after_result['modified_final_levels'])>0:
                modified_after_levels = after_result['modified_final_levels'][-1]


        if auto_retry>0 and time.time()-self.t0<60 and int(after_levels['general_level'])!=target_level and (modified_after_levels is None or int(modified_after_levels['final_levels']['general_level'])!=target_level):
            return self.start_adapt_cefr(text, target_level, target_adjustment=target_adjustment, even=even, n=n, by='piece', auto_retry=auto_retry-1)

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
                if s['CEFR_vocabulary'][i]>threshold:
                    tagged_text += "<b>"+s["word"][i]+"</b>"+' '*s["whitespace"][i]
                    not_tagged = False
                elif s['CEFR_vocabulary'][i]==threshold and (not_tagged or np.random.rand()<p):
                    tagged_text += "<b>"+s["word"][i]+"</b>"+' '*s["whitespace"][i]
                    not_tagged = False
                else:
                    tagged_text += s["word"][i]+' '*s["whitespace"][i]
        return tagged_text
    
    def tag_difficult_sentences(self, result, target_level, target_adjustment=0.5):
        not_tagged = True
        tagged_text = ''
        max_length = self.analyser.cefr2.cefr2length(target_level+target_adjustment+0.5)
        for s in result['sentences'].values():
            length = sum([1 for x in s['pos'] if x not in ['PUNCT','SPACE']])
            sent = ''
            for i in range(len(s['lemma'])):
                sent += s["word"][i]+' '*s["whitespace"][i]
            if int(max(s['CEFR_clause']))>target_level+1 or length>max_length:
                tagged_text += "<i>"+sent+"</i>"
                not_tagged = False
            elif (int(max(s['CEFR_clause']))==target_level+1 or length==max_length) and (not_tagged or np.random.rand()<0.5):
                tagged_text += "<i>"+sent+"</i>"
                not_tagged = False
            else:
                tagged_text += sent
        return tagged_text

    def get_best_candidate(self, candidates, target_level, target_adjustment=0.5, tolerance=50, on='general_level',model='cefr'):
        min_difference = 10000
        finalist_result = None
        if model=='catile':
            for candidate in candidates:
                result = self.analyser.analyze_catile(candidate,v=2)
                if int(round(result['scores']['catile']/10)*10)==target_level:
                    finalist = candidate
                    finalist_result = result
                    break
                else:
                    diff = abs(result['scores']['catile']-target_level)
                    if diff<min_difference:
                        finalist = candidate
                        finalist_result = result
                        min_difference = diff
        else:
            for candidate in candidates:
                result = self.analyser.analyze_cefr(candidate,outputs=['sentences','wordlists','final_levels','modified_final_levels'], v=2)
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

    async def get_adaptation_candidates(self, text, target_level, target_adjustment=0.5, tolerance=50, n=1, change='vocabulary', direction="down", model='cefr'):
        print('Adapting piece')
        n = min(n,10)
        model_name = 'gpt-4-0125-preview'
        n_per_call = min(max(1,int(3000/len(text.split(' '))-1)),n)
        n_self_try = 3
        candidates = []

        if model=='catile':
            if self.example_texts is None:
                catile_examples = df_catile_examples.copy()
                catile_examples['diff'] = catile_examples['catile']-target_level
                catile_examples['diff_abs'] = abs(catile_examples['catile']-target_level)
                catile_examples = catile_examples.sort_values(['diff_abs','diff'])
                self.example_texts = catile_examples.iloc[:5].sample(frac=1)['text'].tolist()+catile_examples.iloc[3:]['text'].tolist()

            while n_self_try>0 and len(candidates)<n:
                n_self_try -= 1
                prompt = self.construct_prompt(text=text, target_level=target_level, example_text=self.example_texts.pop(0), direction=direction, model=model)
                try:
                    if n_self_try==0:
                        completion = await self.async_client.chat.completions.create(
                            model='gpt-4', n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        completion = await self.async_client.chat.completions.create(
                            model=model_name, n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    for x in completion.choices:
                        x = x.message.content.strip()
                        candidates.append(x)
                except Exception as e:
                    if n_self_try==0:
                        self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                        return
                    print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)
        else:
            prompt = self.construct_prompt(text=text, target_level=target_level, target_adjustment=target_adjustment, change=change, direction=direction, model=model)
            while n_self_try>0 and len(candidates)<n:
                n_self_try -= 1
                try:
                    if n_self_try==0:
                        completion = await self.async_client.chat.completions.create(
                            model='gpt-4', n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        completion = await self.async_client.chat.completions.create(
                            model=model_name, n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    for x in completion.choices:
                        x = x.message.content.strip()
                        x = x.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','')
                        candidates.append(x)
                except Exception as e:
                    if n_self_try==0:
                        self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                        return
                    print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)
        return candidates

    def gather_candidates(self, piece_infos_dict):
        keys = []
        values = []
        tasks = []
        for k, v in piece_infos_dict.items():
            keys.append(k)
            values.append({k2:v2 for k2,v2 in v.items() if k2!='stage'})
            tasks.append(self.get_adaptation_candidates(**values[-1]))
        openai_t0 = time.time()
        candidates = asyncio.run(asyncio.gather(*tasks))
        self.openai_time += time.time()-openai_t0
        if values[0]['model']=='catile':
            candidates_infos = [{'candidates':candidates[i], 'target_level':values[i]['target_level'], 'tolerance':values[i]['tolerance'], 'model':'catile'} for i in range(len(values))]
        else:
            candidates_infos = []
            for i in range(len(values)):
                piece_info = values[i]
                if piece_info['direction']=="down":
                    if piece_info['change']!='vocabulary':
                        on = 'clause_level'
                    else:
                        on = 'vocabulary_level'
                else:
                    on = 'general_level'
                candidates_infos.append({'candidates':candidates[i], 'target_level':values[i]['target_level'], 'target_adjustment':values[i]['target_adjustment'], 'on':on, 'model':'cefr'})
        candidates_infos_dict = dict(zip(keys, candidates_infos))
        return candidates_infos_dict
    
    def get_finalists(self, candidates_infos_dict):
        finalists_dict = {}
        for k, v in candidates_infos_dict.items():
            finalists_dict[k] = self.get_best_candidate(**v)
        return finalists_dict

    def get_adaptation(self, text, target_level, target_adjustment=0.5, tolerance=50, n=1, change='vocabulary', direction="down", model='cefr'):
        print('Adapting piece')
        n = min(n,10)
        model_name = 'gpt-4-0125-preview'
        n_per_call = min(max(1,int(3000/len(text.split(' '))-1)),n)
        n_self_try = 3
        candidates = []

        if model=='catile':
            if self.example_texts is None:
                catile_examples = df_catile_examples.copy()
                catile_examples['diff'] = catile_examples['catile']-target_level
                catile_examples['diff_abs'] = abs(catile_examples['catile']-target_level)
                catile_examples = catile_examples.sort_values(['diff_abs','diff'])
                self.example_texts = catile_examples.iloc[:5].sample(frac=1)['text'].tolist()+catile_examples.iloc[3:]['text'].tolist()

            while n_self_try>0 and len(candidates)<n:
                n_self_try -= 1
                prompt = self.construct_prompt(text=text, target_level=target_level, example_text=self.example_texts.pop(0), direction=direction, model=model)
                #print(prompt)
                try:
                    openai_t0 = time.time()
                    if n_self_try==0:
                        completion = self.client.chat.completions.create(
                            model='gpt-4', n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=model_name, n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    self.openai_time += time.time()-openai_t0
                    for x in completion.choices:
                        x = x.message.content.strip()
                        candidates.append(x)
                except Exception as e:
                    if n_self_try==0:
                        self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                        return
                    print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

            finalist, finalist_result = self.get_best_candidate(candidates, target_level, tolerance=tolerance, model=model)

        else:
            prompt = self.construct_prompt(text=text, target_level=target_level, target_adjustment=target_adjustment, change=change, direction=direction, model=model)
            while n_self_try>0 and len(candidates)<n:
                n_self_try -= 1
                try:
                    openai_t0 = time.time()
                    if n_self_try==0:
                        completion = self.client.chat.completions.create(
                            model='gpt-4', n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    else:
                        completion = self.client.chat.completions.create(
                            model=model_name, n=max(n_per_call,n-len(candidates)),
                            messages=[{"role": "user", "content": prompt}]
                        )
                    self.openai_time += time.time()-openai_t0
                    for x in completion.choices:
                        x = x.message.content.strip()
                        x = x.replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','')
                        candidates.append(x)
                except Exception as e:
                    if n_self_try==0:
                        self.result = {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                        return
                    print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

            if direction=="down":
                if change!='vocabulary':
                    on = 'clause_level'
                else:
                    on = 'vocabulary_level'
            else:
                on = 'general_level'
            finalist, finalist_result = self.get_best_candidate(candidates, target_level, target_adjustment=target_adjustment,on=on,model=model)
        
        return finalist, finalist_result

    
    def construct_prompt(self, text, target_level, target_adjustment=0.5, change='vocabulary', example_text='', direction='down', model='cefr'):
        if model=='catile':
            if target_level>1360:
                grade_prompt='university level'
            elif target_level<0:
                grade_prompt='pre-school level'
            else:
                grade2catile = np.array([-160,165,425,645,850,950,1030,1095,1155,1205,1250,1295,1295])
                grade = np.abs(grade2catile-target_level).argmin()
                grade_prompt = f'{grade+5}-year-old'

            prompt = "You are a professional writer for English graded readers. "
            if direction=='down':
                prompt += f"You will be given a text that is too difficult for Lexile {target_level}L readers who are {grade_prompt} kids. "
            else:
                prompt += f"You will be given a text that is too easy for Lexile {target_level}L readers who are {grade_prompt} kids. "
            prompt += f'''You will also be given an excerpt from an example text which is typically for {grade_prompt} kids. '''
            if target_level<500:
                prompt += f'''Your task is to use the repetitive sentence pattern in the example text to write about the ideas in the original text. '''
            else:
                prompt += f'''Your task is to take the main ideas of the original text and use them to write a new text at Lexile {target_level}L. '''
                
            prompt += f'''The difficulty of the vocabulary and sentence patterns in the new text should be similar to that in the example text.
Do not add titles or subtitles if there aren't any in the original text.
            
Original text:
```
{text}
```

Excerpt of an example text:
```
{example_text}
```

Output only the new text without any comments or tags.'''
            return prompt
        else:
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
                    max_length = self.analyser.cefr2.cefr2length(target_level+target_adjustment+0.5)
                    min_length = self.analyser.cefr2.cefr2length(target_level+target_adjustment-0.5)
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
                max_length = self.analyser.cefr2.cefr2length(target_level+target_adjustment+0.5)
                min_length = self.analyser.cefr2.cefr2length(target_level+target_adjustment-0.5)
                prompt += f'''\nThis text is too easy for CEFR {level} level. Rewrite it so that it satisfies these requirements:
    1. There are many words at CEFR {level} level, but no words above CEFR {level} level.
    2. Each sentence should have approximately {min_length} to {max_length} words.

    '''
            prompt += '''
    Do not remove titles and subtitles if there are any.

    Output only the new text without any comments or tags.'''
            return prompt


    def start_adapt_catile(self, text, target_level, tolerance=50, n=1, min_piece_length=2000, auto_retry=False, previous_best=None):
        if self.before_result is None:
            before_result = self.analyser.analyze_catile(text,v=2)
        else:
            before_result = self.before_result
        before_level = int(round(before_result['scores']['catile']/10)*10)
        after_result = None
        adaptations = []

        pieces = []
        pieces = self.divide_piece(text, min_piece_length=min_piece_length, by='piece')
        n_pieces = len(pieces)
        for k,piece in enumerate(pieces):
            if len(piece.strip())==0:
                adaptations.append(piece)
                continue

            if n_pieces>1:
                piece_result = self.analyser.analyze_catile(piece,v=2)
                piece_level = int(round(piece_result['scores']['catile']/10)*10)
            else:
                piece_level = before_level
                piece_result = before_result

            if abs(piece_level-target_level)<=tolerance:
                adaptations.append(piece)
                continue

            adaptation = piece
            if piece_level<target_level:
                adaptation, after_result = self.get_adaptation(adaptation, target_level, tolerance=tolerance, n=n, direction="up",model='catile')
            else:
                adaptation, after_result = self.get_adaptation(adaptation, target_level, tolerance=tolerance, n=n, direction="down",model='catile')
            adaptations.append(adaptation.strip('\n'))

        after_text = ' '.join(adaptations)
        
        if len(adaptations)>1:
            after_result = self.analyser.analyze_catile(after_text,v=2)
        if after_result is None:
            after_level = before_level
        else:
            after_level = after_result['scores']['catile']

        if previous_best is None or abs(after_level-target_level)<abs(previous_best['after']-target_level):
            previous_best = {'adaptation':after_text,'before':before_level,'after': after_level}

        if auto_retry>0 and time.time()-self.t0<90 and abs(int(round(previous_best['after']/10)*10)-target_level)>tolerance:
            return self.start_adapt_catile(text, target_level, tolerance=min(tolerance+10,100), n=n, auto_retry=auto_retry-1, previous_best=previous_best)

        self.result = previous_best


def float2cefr(num):
    cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
    output = cefr.get(int(num),"Native")
    if num<6:
        s = "%.1f" % num
        output += s[s.index('.'):]
    return output