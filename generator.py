import numpy as np
import json, ast, warnings, os, sys, re, difflib, httpx, copy, itertools, pickle
from openai import OpenAI
from nltk.tokenize import sent_tokenize
from .utils import InformError, clean_target_level_input

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
df_us = pickle.load(open(os.path.join(BASE_DIR, 'files/model_files/pronunciation_table_us.pkl'),'rb'))

def parse_response(response):
    try:
        square = response.index('[')
    except:
        square = None
    try:
        curly = response.index('{')
    except:
        curly = None

    sign_l = '['
    sign_r = ']'
    if square is not None and curly is not None:
        if curly<square:
            sign_l = '{'
            sign_r = '}'
    elif square is None and curly is not None:
        sign_l = '{'
        sign_r = '}'

    try:
        response = response[response.index(sign_l):response.rfind(sign_r)+1]
        questions = ast.literal_eval(response)
    except:
        try:
            response = response[response.index(sign_l):response.rfind(sign_r)+1]
            questions = json.loads(response)
        except:
            return None
    return questions

level_str2int = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
level_int2str = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}

class AdoQuestionGenerator(object):
    def __init__(self, text_analyser=None, video_analyser=None, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.client = None
        self.analyser = text_analyser
        self.video_analyser = video_analyser
        
    def generate_questions(self, text=None, url=None, words=None, n=10, kind='multiple_choice', skill='reading', level=None, 
                           answer_position=False, explanation=False, question_language=None, explanation_language=None, auto_retry=3, override_messages=None,
                           transcribe=False, duration_limit=900):
        
        n = min(int(n),20)
        auto_retry = min(int(auto_retry),3)

        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        elif self.client is None:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(120, connect=5))

        if not text and not url:
            raise InformError("Please provide the text or the URL.")
        
        video_result = None
        if text is None and url is not None:
            video_result = self.video_analyser.analyze_youtube_video(url, transcribe=transcribe, auto_transcribe=True, duration_limit=duration_limit)
            text = video_result['video_info']['text']
            if not text:
                raise InformError("Error loading video.")

        if level is None:
            if self.analyser is not None and (question_language is None or question_language=='English'):
                result = self.analyser.analyze_cefr(text,v=2)
                level = min(int(result["final_levels"]["general_level"]),5)
                print(f"Level detected: {level}")
        else:
            level = clean_target_level_input(level)
        
        if level is not None:
            target_level = level_int2str[level]
            max_length = self.analyser.cefr2.cefr2length(level+0.5)
            min_length = self.analyser.cefr2.cefr2length(level-0.5)
            if kind!='multiple_choice_cloze':
                level_prompt = f"The exercises are for CEFR {target_level} students. In the questions and answers,"
            else:
                level_prompt = f"The exercises are for CEFR {target_level} students. In the questions and answers, each sentence should have no more than {max_length} words."
            if level<=2:
                level_prompt += f" You should only use simple vocabulary and sentence structure below {target_level} level so that a child can understand. Don't use technical and academic words."
            elif level<=4:
                level_prompt += f" You should only use vocabulary strictly below {target_level} level."
            else:
                level_prompt = ''
            if level_prompt!='':
                level_prompt += ''' If you fail to stick to these difficulty rules, the students will not understand the questions, and your boss will be angry and punish you.'''
        else:
            level_prompt = ''

        question_language_promt = ''
        if question_language is not None and question_language!='English':
            question_language_promt = 'The questions should be created in '+question_language+'.'
            level_prompt = ''

        explanation_language_promt = ''
        if explanation==True and explanation_language is not None:
            explanation_language_promt = 'The explanation should be written in '+explanation_language+'.'

        if skill=='listening':
            material = 'as a listening exercise for an audio recording'
            material_format = 'Transcript'
        else:
            material = 'for a text'
            material_format = 'Text'

        if answer_position==True:
            answer_position_prompt = 'The question should come with the portion of the original text that indicates the answer ("answer_position").'
            answer_position_json = ', "answer_position": the part of the text that indicates the answer'
        else:
            answer_position_prompt = ''
            answer_position_json = ''

        if explanation==True:
            explanation_prompt = 'The question should come with a short explanation of the answer ("explanation").'
            explanation_json = ', "explanation": explanation of the answer'
        else:
            explanation_prompt = ''
            explanation_json = ''

        format_type = 'list of dictionaries'

        if kind=='multiple_choice':
            if text is None:
                raise InformError("Please provide the text.")
            json_format = '''[{"question": "Why is this the case?", "choices": ["Some choice","Some choice","Some choice","Some choice"], "answer_index": 0***answer_position_json******explanation_json***}, {"question": "What is this?", "choices": ["Some choice","Some choice","Some choice","Some choice"], "answer_index": 2***answer_position_json******explanation_json***}]'''
            json_format = json_format.replace('***answer_position_json***',answer_position_json)
            json_format = json_format.replace('***explanation_json***',explanation_json)

            content = f'''Your task is to generate high-order thinking multiple choice questions {material}. Each question has only one correct choice and three unambiguously incorrect choices. {level_prompt} {answer_position_prompt} {explanation_prompt} {question_language_promt} {explanation_language_promt}
            Follow the steps:
            1. Generate a high-order thinking question with 4 choices. One choice is logically correct and the other three are unambiguously incorrect.
            2. Verify each choice with the text to see if it can be the answer to the question.
            3. If more than one choices are a possible answer to the question, discard this question and start from the beginning.
            4. Repeat this process until you have {n} different questions.

            After you generate {n} questions, arrange them as a Python list of dictionaries in this format:
            ```{json_format}```

            Each dictionary must meet the following requirements:
            1. Each dictionary is one question.
            2. The answer_index ranges from 0 to 3. The distribution of answer_index values (0, 1, 2, 3) should be balanced.
            3. It can be parsed using ast.literal_eval in Python.


            {material_format}:
            ```{text}```
            '''
        elif kind=='essay_question' or kind=='short_answer':
            if text is None:
                raise InformError("Please provide the text.")
            json_format = '''[{"question": "Why is this the case?", "answer": "Some answer"***answer_position_json******explanation_json***}, {"question": "What is this?", "answer": "Some answer"***answer_position_json******explanation_json***}]'''
            json_format = json_format.replace('***answer_position_json***',answer_position_json)
            json_format = json_format.replace('***explanation_json***',explanation_json)

            content = f'''Your task is to generate high-order thinking short answer questions {material}. Each question has only one correct answer. {level_prompt} {answer_position_prompt} {explanation_prompt} {question_language_promt} {explanation_language_promt}

            Answer rules:
            1. Use short phrases when possible. For example, the answer to "What are the freshwater forms of algae called?" should be "Charophyta." instead of "The freshwater forms of algae are called Charophyta."
            2. The answer should have a full stop at the end. For example, "Charophyta." instead of "Charophyta".
            
            Follow the steps:
            1. Generate a high-order thinking question.
            2. Answer the question yourself following the answer rules.
            3. Verify the answer in the text again.
            4. If the answer is not in the text, or if the answer is contradictory or ambiguous, discard this question and start from the beginning.
            5. Repeat this process until you have {n} different questions.

            After you generate {n} questions, arrange them as a Python list of dictionaries in this format:
            ```{json_format}```

            Each dictionary must meet the following requirements:
            1. Each dictionary is one question.
            2. It can be parsed using ast.literal_eval in Python.


            {material_format}:
            ```{text}```
            '''

        elif kind=='true_false' or kind=='true_false_not_given':
            if text is None:
                raise InformError("Please provide the text.")
            json_format = '''[{"question": statement 1, "answer": False***answer_position_json******explanation_json***}, {"question": statement 2,"answer": True***answer_position_json******explanation_json***}, ...]'''
            json_format = json_format.replace('***answer_position_json***',answer_position_json)
            json_format = json_format.replace('***explanation_json***',explanation_json)

            if kind=='true_false_not_given':
                question_type = 'True/False/Not Given'
                question_rule2 = '2. The number of the three answers should be balanced. And there should be at least one "Not Given" statement.'
                question_rule3 = '3. For "Not Given" statements, the "answer_position" should be "N/A"'
            else:
                question_type = 'True/False'
                question_rule2 = '2. The number of the two answers should be balanced.'
                question_rule3 = ''
            

            content = f'''Your task is to generate high-order thinking {question_type} questions for {material}. {level_prompt} {answer_position_prompt} {explanation_prompt} {question_language_promt} {explanation_language_promt}

            Question rules:
            1. The statement in the question should not be the original sentence from the text. You should paraphrase and write it with your own words.
            {question_rule2}
            {question_rule3}

            After you generate {n} questions, arrange them as a Python list of dictionaries in this format:
            ```{json_format}```

            Each dictionary must meet the following requirements:
            1. Each dictionary is one question.
            2. It can be parsed using ast.literal_eval in Python.

            {material_format}:
            ```{text}```
            '''

        elif kind=='sentence_completion':
            if text is None and words is None:
                raise InformError("Please provide the text or the words.")

            if words is not None:
                if isinstance(words, str) and text is not None:
                    content = f'''Your task is to generate sentence completion exercises for the words wrapped in {words} in the {material_format}'''
                else:
                    content = f'''Your task is to generate sentence completion exercises for these words: {', '.join(words)}.'''
            else:
                content = f'''Your task is to generate {n} sentence completion exercises for the important words in the {material_format}.'''

            content += f''' {level_prompt} {answer_position_prompt} {explanation_prompt} {explanation_language_promt}'''

            if text is not None:
                content += ''' The sentence contexts should be similar to the original settings but should not be the same as the original. The meaning and part of speech of the words in the sentences should be the same as the original.\n\n'''
                content += f'''{material_format} and contexts of the words:\n```{text}```\n\n'''

            json_format = '''[{"sentence": "sentence with a blank", "answer": "the word"***answer_position_json******explanation_json***}, {"sentence": "sentence with a blank", "answer": "the word"***answer_position_json******explanation_json***}, ...]'''
            json_format = json_format.replace('***answer_position_json***',answer_position_json)
            json_format = json_format.replace('***explanation_json***',explanation_json)
            content += f''' Arrange the sentences as a Python list of dictionaries in this format:
            ```{json_format}```
            Each dictionary must meet the following requirements:
            1. Each dictionary is one sentence.
            2. It can be parsed using ast.literal_eval in Python.
            '''

        elif kind=='word_matching':
            if words is not None:
                content = f'''Your task is to generate definition match exercises for these words: {', '.join(words)}.'''
            else:
                content = f'''Your task is to generate {n} definition match exercises for the important words in the {material_format}.'''

            content += f''' {answer_position_prompt} {explanation_prompt} {question_language_promt} '''

            if text is not None:
                content += '''The meaning and part of speech of the words should be the same as in the contexts.\n'''

            if question_language is not None:
                content += f'''The definition should be the direct, one-word translations of the words in {question_language} with no English explanations.\n'''
            elif level_prompt!='':
                content += level_prompt

            if text is not None:
                content += f'''\n{material_format} and contexts of the words:\n```{text}```\n\n'''

            json_format = '''[{"definition": definition1, "answer": the word***answer_position_json******explanation_json***}, {"definition": definition2, "answer": the word***answer_position_json******explanation_json***}, ...]'''
            json_format = json_format.replace('***answer_position_json***',answer_position_json)
            json_format = json_format.replace('***explanation_json***',explanation_json)
            content += f''' Arrange the sentences as a Python list of dictionaries in this format:
            ```{json_format}```
            Each dictionary must meet the following requirements:
            1. Each dictionary is one sentence with keys "definition" and "answer".
            2. It can be parsed using ast.literal_eval in Python.
            '''

        elif kind=='multiple_choice_cloze':
            if text is None and words is None:
                raise InformError("Please provide the text or the words.")

            requirements = '''Each question should have four choices. The blank in the text should have the number of the question. For example, if the blank is the first question, the blank should be ___1___; If the blank is the second question, the blank should be ___2___, etc.'''

            if words is not None:
                if isinstance(words,str):
                    actual_words = re.findall(r'\*\*\*([^\*]+)\*\*\*', text)
                    content = f'''Your task is to generate multiple choice cloze exercises for the words wrapped in {words} in the {material_format}, and these words are [{', '.join(actual_words)}]. {requirements} '''
                else:
                    content = f'''Your task is to generate multiple choice cloze exercises for these words: [{', '.join(words)}] in the {material_format}. {requirements} '''
            else:
                content = f'''Your task is to generate multiple choice cloze exercises for {n} important words in the {material_format}. The words should test students' vocabulary and grammar instead of factual knowledge, so don't choose numbers or proper nouns. {requirements} '''

            content += f''' {level_prompt} {explanation_prompt} {explanation_language_promt}'''

            if text is not None:
                content += f'''{material_format}:\n```{text}```\n\n'''

            json_format = '''{"text": text with blanks, questions:[{"choices": ["Some choice","Some choice","Some choice","Some choice"], "answer_index": 0***explanation_json***}, {"question": "What is this?", "choices": ["Some choice","Some choice","Some choice","Some choice"], "answer_index": 2***explanation_json***}, ...]}'''
            json_format = json_format.replace('***explanation_json***',explanation_json)
            format_type = 'dictionary'
            content += f'''Arrange the text with blanks and questions as a Python dictionary in this format:
            ```{json_format}```

            The output dictionary must meet the following requirements:
            1. The answer words must be replaced by blanks in the text, and the correct answer must be exactly the same as the words that were replaced.
            2. The text with blanks will be the value of the 'text' key. The paragraph format and new lines should be the same as the original text.
            3. "questions" is a list of dictionaries. Each dictionary is one question with "choices", and "answer_index".
            4. The answer_index ranges from 0 to 3. The distribution of answer_index values (0, 1, 2, 3) should be balanced.
            5. Excape new lines for the value of "text" because there may be line breaks.
            6. It can be parsed using ast.literal_eval in Python.
            '''

        messages = [{"role": "user", "content": content}]
        
        if override_messages is None:
            messages_to_send = messages
        else:
            messages_to_send = override_messages

        n_self_try = 3
        while n_self_try>0:
            try:
                completion = self.client.chat.completions.create(
                    model='gpt-4-1106-preview',#"gpt-3.5-turbo",
                    messages=messages_to_send
                )
                break
            except Exception as e:
                n_self_try -= 1
                if n_self_try==0:
                    return {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

        response = completion.choices[0].message.content.strip(' `')
        questions = parse_response(response)

        if questions is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.generate_questions(text=text, url=url, n=n, kind=kind, auto_retry=auto_retry-1, words=words, skill=skill, level=level, answer_position=answer_position, explanation=explanation, question_language=question_language, explanation_language=explanation_language, 
                                                   override_messages=messages+[{"role": completion.choices[0].message.role, "content": completion.choices[0].message.content},
                                                                               {"role": "user", "content": f"The questions you returned are not in Python {format_type} format. Return them as a Python {format_type} like this example: {json_format}"}])
                else:
                    return self.generate_questions(text=text, url=url, n=n, kind=kind, auto_retry=auto_retry-1, words=words, skill=skill, level=level, answer_position=answer_position, explanation=explanation, question_language=question_language, explanation_language=explanation_language)
            else:
                print("The bot didn't return the questions in Python dictionary format. Response: {response}")
                raise InformError("Creating questions failed. Please try again or use a different text.")
        elif len(questions)<n and auto_retry>0:
            return self.generate_questions(text=text, url=url, n=n, kind=kind, auto_retry=auto_retry-1, words=words, skill=skill, level=level, answer_position=answer_position, explanation=explanation, question_language=question_language, explanation_language=explanation_language)
        
        if kind=='essay_question':
            for i in range(len(questions)):
                questions[i]['answer'] = questions[i]['answer'].capitalize()
        if isinstance(questions,list) and len(questions)>n:
            questions = questions[:n]

        if isinstance(questions,list):
            result = {"questions":questions}
        else:
            result = questions
        if video_result:
            result.update(video_result)
        return result
    
    def parse_questions(self, response):
        try:
            square = response.index('[')
        except:
            square = None
        try:
            curly = response.index('{')
        except:
            curly = None

        sign_l = '['
        sign_r = ']'
        if square is not None and curly is not None:
            if curly<square:
                sign_l = '{'
                sign_r = '}'
        elif square is None and curly is not None:
            sign_l = '{'
            sign_r = '}'

        try:
            response = response[response.index(sign_l):response.rfind(sign_r)+1]
            questions = ast.literal_eval(response)
        except:
            try:
                response = response[response.index(sign_l):response.rfind(sign_r)+1]
                questions = json.loads(response)
            except:
                return None
        return questions
    
    

class AdoTextGenerator(object):
    def __init__(self, text_analyser, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.analyser = text_analyser

        self.grammar_description = {'present simple passive':'is/are (not) done',
                                    'present continuous':'is/are (not) doing',
                                    'present continuous passive':'is/are (not) being done',
                                    'present perfect':'have/has (not) done',
                                    'present perfect continuous':'have/has (not) been doing',
                                    'present perfect passive':'have/has (not) been done',
                                    'past simple passive':'was/were (not) done',
                                    'past continuous':'was/were (not) doing',
                                    'past continuous passive':'was/were (not) being done',
                                    'past perfect':'had (not) done',
                                    'past perfect continuous':'had (not) been doing',
                                    'past perfect passive':'had (not) been done',
                                    'gerund passive':'verb/preposition + being done',
                                    'gerund perfect':'verb/preposition + having done',
                                    'gerund perfect continuous':'verb/preposition + having been doing',
                                    'gerund perfect passive':'verb/preposition + having been done',
                                    'infinitive passive':'(not) to be done, can/could/should/... (not) be done',
                                    'infinitive continuous':'(not) to be doing, can/could/should/... (not) be doing',
                                    'infinitive perfect':'(not) to have done, can/could/should/... (not) have done',
                                    'infinitive perfect continuous':'(not) to have been doing, can/could/should/... (not) have been doing',
                                    'infinitive perfect passive':'(not) to have been done, can/could/should/... (not) have been done'}

    def create_text(self,level,n_words=300,topic=None,keywords=None,grammar=None,genre=None,
                    settings={'ignore_keywords':True,'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'custom_dictionary':{}},
                    outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels','modified_final_levels']):
        
        if settings.get('search') is not None:
            if settings['search']['function'] == 'search_words':
                return {"words":self.search_words(**settings['search']['parameters'])}
            elif settings['search']['function'] == 'quick_wordlist':
                return {"words":self.quick_wordlist(**settings['search']['parameters'])}
            elif settings['search']['function'] == 'get_minimal_pairs':
                return {"words":self.get_minimal_pairs(**settings['search']['parameters'])}
            elif settings['search']['function'] == 'get_blends':
                return {"words":self.get_blends(**settings['search']['parameters'])}
        
        if grammar is not None and not isinstance(grammar,list):
            raise InformError("grammar must be a list of strings.")
        if keywords is not None and not isinstance(keywords,list):
            raise InformError("keywords must be a list of strings.")

        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(150, connect=5))

        level = clean_target_level_input(level)

        prompt = self.construct_prompt(level=level,n_words=n_words,topic=topic,keywords=keywords,grammar=grammar,genre=genre)

        custom_dictionary = {}
        if keywords and settings.get('ignore_keywords',True):
            for x in keywords:
                if isinstance(x,str):
                    custom_dictionary[x] = -1
                else:
                    custom_dictionary[tuple(x)] = -1
            custom_dictionary.update(settings.get('custom_dictionary',{}))
        default_settings = {'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'custom_dictionary':{}}
        default_settings.update(settings)
        if 'ignore_keywords' in default_settings:
            del default_settings['ignore_keywords']
        settings = default_settings
        settings['custom_dictionary'] = custom_dictionary
        if grammar:
            model = 'gpt-4'
        else:
            model = 'gpt-3.5-turbo'

        return self.execute_prompt(prompt,level,temp_results=[], model=model, settings=settings, outputs=outputs)

    def construct_prompt(self, level,n_words=300,topic=None,keywords=None,grammar=None,genre=None):

        target_level = level_int2str[level]
        max_length = self.analyser.cefr2.cefr2length(level+0.5)
        min_length = self.analyser.cefr2.cefr2length(level-0.5)
        requirements = ['There should not be a title.']

        if topic:
            requirements.append(f"The topic is {topic}.")
        if genre:
            requirements.append(f"The genre (or type of text) is {genre}.")
        if keywords:
            keywords_to_join = []
            for x in keywords:
                if isinstance(x,str):
                    keywords_to_join.append(x)
                else:
                    keywords_to_join.append(f'{x[0]} ({x[1]})')
            requirements.append(f"It should include these words: {', '.join(keywords_to_join)}.")
        if grammar:
            grammar_list = ''
            for x in grammar:
                grammar_list += '\t'+x
                if x.lower() in self.grammar_description:
                    grammar_list += ': '+self.grammar_description[x.lower()]+';\n'
                else:
                    grammar_list += ';\n'
            requirements.append("Use these grammar structures many times:\n"+grammar_list)
        
        requirements.append(f"It should be around {n_words} words.")
        requirements.append("Use proper paragraphing to separate and organise the content into a few paraghraphs. Don't just write everything in one single paragraph.")
        requirements.append('''Arrange the result in a python dictionary like this:\n```{"text": the text}```\n Excape new lines for the value of "text" because there may be line breaks. It should be parsed directly. Don't include other notes, tags or comments.''')
        
        if level<=2:
            prompt = f'''
You task is to write an English text at CEFR {target_level} level for elementary English learners. The difficulty of vacubalary is important. You must choose your words carefully and use only simple words. Don't use technical or academic vocabulary.

{target_level} level texts should meet these requirements:
1. Each sentence has no less than {min_length} words and no more than {max_length} words.
2. The vocabulary should be simple and below CEFR {target_level} level.

In the meantime, the text should meet the following requirements:
'''
        else:
            prompt = f'''
You task is to write an English {genre} text at CEFR {target_level} level.

{target_level} level texts should meet these requirements:
1. Each sentence has {min_length} to no more than {max_length} words.
2. The vocabulary should not be more difficult than CEFR {target_level} level.

In the meantime, the text should meet the following requirements:
'''

        for i, x in enumerate(requirements):
            prompt += f"{i+1}. {x}\n"
        
        return prompt.strip('\n').replace(' None ',' ')

    def execute_prompt(self,prompt,level,auto_retry=3,temp_results=[],model='gpt-3.5-turbo',
                       settings={'propn_as_lowest':True,'intj_as_lowest':True,'keep_min':True,'custom_dictionary':{}},
                       outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels','modified_final_levels']):
        model = 'gpt-4-1106-preview'
        n_trials = len(temp_results)+1
        print(f"Trying {n_trials}")
        for i in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1
                )
                text = parse_response(completion.choices[0].message.content)['text'].strip()
                break
            except Exception as e:
                print(completion.choices[0].message.content)
                if i==2:
                    raise e
                continue
        result = self.analyser.analyze_cefr(text,settings=settings,outputs=outputs,v=2)

        if int(result['final_levels']['general_level'])!=level:
            if auto_retry>0:
                if text:
                    temp_results.append([result['final_levels']['general_level'],text,result])
                #if len(temp_results)>=2:
                #    return self.execute_prompt(prompt,max(0,level-1),auto_retry=auto_retry-1,temp_results=temp_results)
                #else:
                #    return self.execute_prompt(prompt,level,auto_retry=auto_retry-1,temp_results=temp_results)
                return self.execute_prompt(prompt,level,auto_retry=auto_retry-1,temp_results=temp_results)
            else:
                diffs = []
                for i in range(len(temp_results)):
                    diffs.append(abs(temp_results[i][0]-level))
                best_i = np.argmin(diffs)
                text = re.sub(r'\([0-9]+\)', '', temp_results[best_i][1]).replace('Title: ','').replace('Text: ','').replace('  ',' ')
                lines = text.split('\n')
                if lines[0].startswith('"') and lines[0].endswith('"'):
                    lines[0] = lines[0][1:-1]
                    text = '\n'.join(lines)
                result = temp_results[best_i][2]
                return {'text':text, 'result':result}
        else:
            text = re.sub(r'\([0-9]+\)', '', text).replace('Title: ','').replace('Text: ','').replace('  ',' ')
            lines = text.split('\n')
            if lines[0].startswith('"') and lines[0].endswith('"'):
                lines[0] = lines[0][1:-1]
                text = '\n'.join(lines)
            return {'text':text, 'result':result}

    

    def search_words(self,phonemes,pos=None,n_syllables=None,cefr=None,n=10,ignore_stress=True):

        df_temp = df_us.copy()
        
        if isinstance(n_syllables,int):
            df_temp = df_temp[df_temp['n_syllables']==n_syllables]
        elif n_syllables:
            df_temp = df_temp[(df_temp['n_syllables']>=n_syllables[0])&(df_temp['n_syllables']<=n_syllables[1])]
        if isinstance(cefr,int):
            df_temp = df_temp[df_temp['cefr']==cefr]
        elif isinstance(cefr,str):
            cefr = level_str2int[cefr]
            df_temp = df_temp[df_temp['cefr']==cefr]
        elif cefr:
            cefr = [level_str2int.get(x,x) for x in cefr]
            df_temp = df_temp[df_temp['cefr'].apply(lambda x: x in cefr)]
        if isinstance(pos,str):
            df_temp = df_temp[df_temp['pos']==pos]
        elif isinstance(pos,list):
            df_temp = df_temp[df_temp['pos'].apply(lambda x: x in pos)]

        df_temp = df_temp.sample(frac=1)
        sounds = []
        positions = []
        for phoneme in phonemes:
            if isinstance(phoneme['sound'],str):
                sounds.append([phoneme['sound']])
            else:
                sounds.append(phoneme['sound'])
            positions.append(phoneme.get('position',[None,None]))
        filter_ = []
        n_matched = 0
        for row in df_temp.to_dict('records'):
            matched = True
            for i in range(len(sounds)):
                sound = sounds[i]
                position = positions[i]
                if ignore_stress:
                    arpabet = np.array(row['arpabet_uncased'],dtype='object')
                else:
                    arpabet = np.array(row['arpabet'],dtype='object')

                    
                if isinstance(position[0],int):
                    position[0] = [position[0]]
                if isinstance(position[1],int):
                    position[1] = [position[1]]


                if position[0] is None and position[1] is None:
                    l = arpabet
                elif position[0] is None:
                    l = arpabet[:,position[1]]
                else:
                    position[0] = [x for x in position[0] if x<row['n_syllables']]
                    if len(position[0])==0:
                        matched = False
                        break
                    if position[1] is None:
                        l = arpabet[position[0],:]
                    else:
                        l = arpabet[position[0],position[1]]
                if len(l.shape)==3:
                    l = itertools.chain.from_iterable(l)
                else:
                    l = l.flatten()

                index = np.array(sound)!='@'
                if not any(index):
                    l = [np.array_equal(x,sound) for x in l if len(x)==len(index)]
                else:
                    l = [np.array_equal(np.array(list(x))[index],np.array(sound)[index]) for x in l if len(x)==len(index)]

                matched = len(l)>0 and any(l)
                
                if not matched:
                    break
            filter_.append(matched)
            if matched:
                n_matched += 1
                if n_matched>=n:
                    break
        filter_ += [False]*(len(df_temp)-len(filter_))
        return df_temp[filter_]

    def quick_wordlist(self,phonemes,stress='ignore',position=None, pos=None, n_syllables=None,cefr=None,exclude_spelling=None,n=10):
        if stress=='stressed':
            phonemes = [x.upper() for x in phonemes]
        else:
            phonemes = [x.lower() for x in phonemes]
        
        result = self.search_words([{'sound':phonemes}], ignore_stress=stress=='ignore', pos=pos, cefr=cefr, n_syllables=n_syllables, n=1000000)
        
        if exclude_spelling is not None and len(result)>0:
            if exclude_spelling.startswith('-') and exclude_spelling.endswith('-'):
                exclude_spelling = exclude_spelling.strip('-')
                result = result[result['headword'].apply(lambda x: not x.lower().startswith(exclude_spelling) and not x.lower().startswith(exclude_spelling) and exclude_spelling not in x.lower())]
            elif exclude_spelling.startswith('-') and not exclude_spelling.endswith('-'):
                exclude_spelling = exclude_spelling.strip('-')
                result = result[result['headword'].apply(lambda x: not x.lower().startswith(exclude_spelling))]
            elif not exclude_spelling.startswith('-') and exclude_spelling.endswith('-'):
                exclude_spelling = exclude_spelling.strip('-')
                result = result[result['headword'].apply(lambda x: not x.lower().endsswith(exclude_spelling))]
            else:
                result = result[result['headword'].apply(lambda x: exclude_spelling not in x.lower())]

        if position is not None and len(result)>0:
            phonemes_flatten = '-'.join(phonemes)
            if stress=='ignore':
                arpabet_column = 'arpabet_uncased'
            else:
                arpabet_column = 'arpabet'
            if position=='initial':
                result['flatten'] = result[arpabet_column].apply(lambda x: '-'.join(list(itertools.chain.from_iterable(x[0]))))
                result = result[result['flatten'].apply(lambda x: x.startswith(phonemes_flatten))]
            elif position=='final':
                result['flatten'] = result[arpabet_column].apply(lambda x: '-'.join(list(itertools.chain.from_iterable(x[-1]))))
                result = result[result['flatten'].apply(lambda x: x.endswith(phonemes_flatten))]
            else:
                result['flatten1'] = result[arpabet_column].apply(lambda x: '-'.join([y for y in itertools.chain.from_iterable(x[0]) if y!='']))
                result['flatten2'] = result[arpabet_column].apply(lambda x: '-'.join([y for y in itertools.chain.from_iterable(x[-1]) if y!='']))
                result1 = result[result['flatten1'].apply(lambda x: x.startswith(phonemes_flatten))]
                result2 = result[result['flatten2'].apply(lambda x: x.endswith(phonemes_flatten))]
                result = result.loc[list(set(result.index)-set(result1.index).union(set(result2.index)))]
        
        result = result.sample(min(n,len(result)))
        words = []
        for row in result.to_dict('records'):
            respelling = row['syllables']
            respelling[row['i_stressed']] = respelling[row['i_stressed']].upper()
            words.append({'word':row['headword'],'pronunciation':'-'.join(respelling)})
        return words

    def get_blends(self, phonemes, type='vowel',stress='ignore',n_syllables=None,cefr=None,n=10):
        if stress=='stressed':
            phonemes = [x.upper() for x in phonemes]
        elif stress=='unstressed':
            phonemes = [x.lower() for x in phonemes]

        if type=='initial_consonant':
            result = self.search_words([{'sound':phonemes,'position':[None,0]}],n_syllables=n_syllables,cefr=cefr,ignore_stress=stress=='ignore',n=100000)
        elif type=='final_consonant':
            result = self.search_words([{'sound':phonemes,'position':[None,2]}],n_syllables=n_syllables,cefr=cefr,ignore_stress=stress=='ignore',n=100000)
        else:
            result = self.search_words([{'sound':phonemes,'position':[None,1]}],n_syllables=n_syllables,cefr=cefr,ignore_stress=stress=='ignore',n=100000)
        result = result.sample(min(n,len(result)))
        words = []
        for row in result.to_dict('records'):
            respelling = row['syllables']
            respelling[row['i_stressed']] = respelling[row['i_stressed']].upper()
            words.append({'word':row['headword'],'pronunciation':'-'.join(respelling)})
        return words

    def get_minimal_pairs(self, phonemes, stress="stressed", pos=None, cefr=None, n_syllables=1,n=10):

        def flatten_list(nested_list):
            result = []
            for element in nested_list:
                if isinstance(element, list) or isinstance(element, np.ndarray):
                    # If the element is a list, extend the result by the flattened element
                    result.extend(flatten_list(element))
                else:
                    # If the element is not a list, append it to the result
                    result.append(element)
            return result
        
        if stress=='stressed':
            phonemes = [x.upper() for x in phonemes]
        elif stress=='unstressed':
            phonemes = [x.lower() for x in phonemes]

        if phonemes[0]==phonemes[1]:
            return []
        pairs = []

        
        for i in range(n_syllables):

            df1 = self.search_words([{'sound':[phonemes[0]],'position':[i,None]}], pos=pos, cefr=cefr, ignore_stress=stress=='ignore', n_syllables=n_syllables, n=100000).copy().sample(frac=1)
            df2 = self.search_words([{'sound':[phonemes[1]],'position':[i,None]}], pos=pos, cefr=cefr, ignore_stress=stress=='ignore', n_syllables=n_syllables, n=100000).copy().sample(frac=1)
        
            arpabet_str = []
            for x in df1['arpabet'].values:
                s = copy.deepcopy(x)
                s[i][1] = '@'
                arpabet_str.append('-'.join(flatten_list(s)))
            df1['arpabet_str'] = arpabet_str
            df1 = df1.drop_duplicates('arpabet_str')
        
            arpabet_str = []
            for x in df2['arpabet'].values:
                s = copy.deepcopy(x)
                s[i][1] = '@'
                arpabet_str.append('-'.join(flatten_list(s)))
            df2['arpabet_str'] = arpabet_str
            df2 = df2.drop_duplicates('arpabet_str')
        
            for row in df1.to_dict('records'):
                temp = df2[df2['arpabet_str']==row['arpabet_str']]
                if len(temp)>0:
                    pairs.append([row['headword'],temp.iloc[0]['headword']])
                    if len(pairs)>=n:
                        return pairs
        return pairs


class AdoWritingAssessor(object):
    def __init__(self, text_analyser, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.analyser = text_analyser
        self.client = None

    def revise(self,text, comment=False, writing_language='English', comment_language=None, auto_retry=2, original_analysis=None, override_messages=None):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(150, connect=5))

        if comment==True:
            json_format = '''[{"original": original sentences, "revision": the improved sentences, "type":[revision type, ...], "comment": ...]}, {"original": original sentences, "revision": the improved sentences, "type":[revision type, ...], "comment": ...]}, ...]'''
            comment_prompt = '''You should give comments on the revisions, including vocabulary, spelling errors, grammar mistakes, sentence structures, coherence, etc. Besides the types of revision, also quote the changes to tell students clearly what to improve.'''
            comment_prompt2 = ', and the comments on the revisions'
        else:
            json_format = '''[{"original": original sentences, "revision": the improved sentences, "type":[revision type, ...]]}, {"original": original sentences, "revision": the improved sentences, "type":[revision type, ...]]}, ...]'''
            comment_prompt = ''
            comment_prompt2 = ''

        if comment==True and comment_language:
            language_prompt = f'''Your students speak {comment_language}, so the comments should be in {comment_language}, but keep the quotations of the original and revised text in {writing_language}.'''
        else:
            language_prompt = ''
            
        if original_analysis is None:
            original_analysis = self.analyser.analyze_cefr(text, outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels'],v=2)
        
        target_level = level_int2str.get(np.ceil(original_analysis['final_levels']['vocabulary_level']), 'C2')

        content = f'''You are a professional {writing_language} teacher. Your task is to find mistakes in students' writings and correct them in terms of vocabulary, grammar, sentence structure, paragraphing, and coherence.
For vocabulary and collocations, only correct mistakes and inappropriate usage using words at CEFR {target_level} level or below. Don't try to enhance the vocabulary level.
You can revise one sentence or several sentences at a time when appropriate (for example, when two sentences need to be combined or one sentence needs to be broken down into two).
You can divide, combine or rearrange sentences and paragraphing to improve coherence and cohesion (for example, when a paragraph needs to be divided, add a new line '\\n' at the appropriate position)..
You need to output the original parts of text, the revision, the types of revision{comment_prompt2} in a Python list of dictionaries like this {json_format}
This list of dictionaries should be parsed using ast.literal_eval in Python.
The types of revision can be one or more of the following:
    "vocabulary" (Correct vocabulary usage and collocations mistakes, or correct spelling errors.)
    "grammar"(Correct grammar or sentence structure mistakes, or improve grammatical accuracy.)
    "coherence_cohesion"(Improve coherence and cohesion between sentences and paragraphs by adding or modifying transition words, conjunctions, etc.)
    "others"
{comment_prompt}
If the original sentences are already good, don't include them in the list.
{language_prompt}

If you do your job well by following all the instructions, I will pay you a bonus of $200.

Writing:
{text}
'''
        messages = [{"role": "user", "content": content}]
        if override_messages is None:
            messages_to_send = messages
        else:
            messages_to_send = override_messages

        completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages_to_send
        )

        result = parse_response(completion.choices[0].message.content)
        if result is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.revise(text, comment=comment, writing_language=writing_language, comment_language=comment_language, auto_retry=auto_retry-1, original_analysis=original_analysis, override_messages=messages+[{"role": completion.choices[0].message.role, "content": completion.choices[0].message.content},
                                                                                                              {"role": "user", "content": f"The output you returned are not in the correct Python list of dictionaries format. Return them as a Python list of dictionaries like this example: {json_format}"}])
                else:
                    return self.revise(text, comment=comment, writing_language=writing_language, comment_language=comment_language, auto_retry=auto_retry-1, original_analysis=original_analysis)
            else:
                print(f"The bot didn't return a Python dictionary. Response: {completion.choices[0].message.content}")
                raise InformError("Task failed. Please try again or use a different text.")
        
        revised_text = text+''
        result2 = []
        for x in result:
            if x['original'] in text:
                revised_text = revised_text.replace(x['original'], x['revision'])
            else:
                originals = sent_tokenize(x['original'])
                try:
                    start = revised_text.index(originals[0])
                    end = revised_text.index(originals[-1])+len(originals[-1])
                    revised_text = revised_text[:start]+x['revision']+revised_text[end:]
                except:
                    n_originals = len(originals)
                    for i in range(len(originals)):
                        if i==n_originals-1:
                            revised_text = revised_text.replace(originals[i], x['revision'])
                        else:
                            revised_text = revised_text.replace(originals[i], '')
            diff = self.mark_differences(x['original'],x['revision'])
            x.update({'original_tagged':diff[0],'revision_tagged':diff[1],'combined_tagged':diff[2]})
            result2.append(x)

        revision_analysis = self.analyser.analyze_cefr(revised_text, outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels'],v=2)

        original_analysis['final_levels']['clause_level'] = revision_analysis['final_levels']['clause_level']
        average_level = (original_analysis['final_levels']['vocabulary_level']+original_analysis['final_levels']['tense_level']+original_analysis['final_levels']['clause_level'])/3
        general_level = max([original_analysis['final_levels']['vocabulary_level'],original_analysis['final_levels']['tense_level'],average_level,original_analysis['final_levels']['clause_level']-0.5])
        original_analysis['final_levels']['general_level'] = general_level
        original_analysis['final_levels_str'] = {k:self.analyser.cefr.float2cefr(v) for k,v in original_analysis['final_levels'].items()}

        diff = self.mark_differences(text,revised_text)
        return {'revised_text':revised_text,'revised_text_tagged':diff[1],'original_text_tagged':diff[0],'combined_text_tagged':diff[2],'revisions':result2,'original_analysis':original_analysis, 'revision_analysis':revision_analysis}

    def enhance(self, text, level=None, comment=False, writing_language='English', comment_language=None, auto_retry=2, override_messages=None):
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            self.client = OpenAI(api_key=self.openai_api_key, timeout=httpx.Timeout(150, connect=5))

        if comment==True:
            json_format = '''[{"original": original sentences, "revision": the improved sentences, "type":[revision type, ...], "comment": ...]}, {"original": original sentences, "revision": the improved sentences, "type":[revision type, ...], "comment": ...]}, ...]'''
            comment_prompt = '''You should give comments on the improvements, including vocabulary, spelling errors, grammar mistakes, sentence structures, coherence, etc. Besides the types of revision, also quote the changes to tell students clearly what to improve.'''
            comment_prompt2 = ', and the comments on the revisions'
        else:
            json_format = '''[{"original": original sentences, "revision": the improved sentences, "type":[revision type, ...]]}, {"original": original sentences, "revision": the improved sentences, "type":[revision type, ...]]}, ...]'''
            comment_prompt = ''
            comment_prompt2 = ''

        if comment==True and comment_language:
            language_prompt = f'''Your students speak {comment_language}, so the comments should be in {comment_language}, but keep the quotations of the original and revised text in {writing_language}.'''
        else:
            language_prompt = ''
            
        if level:
            level = clean_target_level_input(level)
            
            if level in level_int2str:
                target_level = level_int2str[level]
                max_length = self.analyser.cefr2.cefr2length(level+0.5)
                min_length = self.analyser.cefr2.cefr2length(level-0.5)
                level_prompt = f"The improved writing should have a level of {target_level}. To enhanced vocabulary, add only several words at {target_level} level. Add no words above {target_level} level. The majority of words should be below {target_level} level."
            else:
                level_prompt = f"The improved writing should have a level of {level}."
        else:
            level_prompt = ''
            
        content = f'''You are a professional {writing_language} teacher. Your task is to improve students' writings in terms of vocabulary, grammar, sentence structure, paragraphing, and coherence. {level_prompt}
You can revise one sentence or several sentences at a time when appropriate (for example, when two sentences need to be combined or one sentence needs to be broken down into two).
You can divide, combine or rearrange sentences and paragraphing to improve coherence and cohesion (for example, when a paragraph needs to be divided, add a new line '\\n' at the appropriate position)..
You need to output the original parts of text, the revision, the types of revision{comment_prompt2} in a Python list of dictionaries like this {json_format}
This list of dictionaries should be parsed using ast.literal_eval in Python.
The types of revision can be one or more of the following:
    "vocabulary" (Enhance vocabulary range and collocations, or correct spelling errors.)
    "grammar"(Correct grammar or sentence structure mistakes, or improve grammatical range and accuracy.)
    "coherence_cohesion"(Improve coherence and cohesion between sentences and paragraphs by adding or modifying transition words, conjunctions, etc.)
    "others"
{comment_prompt}
If the original sentences are already good, don't include them in the list.
{language_prompt}

If you do your job well by following all the instructions, I will pay you a bonus of $200.

Writing:
{text}
'''
        messages = [{"role": "user", "content": content}]
        if override_messages is None:
            messages_to_send = messages
        else:
            messages_to_send = override_messages

        completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages_to_send
        )

        result = parse_response(completion.choices[0].message.content)
        if result is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.enhance(text, level=level, comment=comment, writing_language=writing_language, comment_language=comment_language, auto_retry=auto_retry-1, override_messages=messages+[{"role": completion.choices[0].message.role, "content": completion.choices[0].message.content},
                                                                                                              {"role": "user", "content": f"The output you returned are not in the correct Python list of dictionaries format. Return them as a Python list of dictionaries like this example: {json_format}"}])
                else:
                    return self.enhance(text, level=level, comment=comment, writing_language=writing_language, comment_language=comment_language, auto_retry=auto_retry-1)
            else:
                print(f"The bot didn't return a Python dictionary. Response: {completion.choices[0].message.content}")
                raise InformError("Task failed. Please try again or use a different text.")
        revised_text = text+''
        result2 = []
        for x in result:
            if x['original'] in text:
                revised_text = revised_text.replace(x['original'], x['revision'])
            else:
                originals = sent_tokenize(x['original'])
                try:
                    start = revised_text.index(originals[0])
                    end = revised_text.index(originals[-1])+len(originals[-1])
                    revised_text = revised_text[:start]+x['revision']+revised_text[end:]
                except:
                    n_originals = len(originals)
                    for i in range(len(originals)):
                        if i==n_originals-1:
                            revised_text = revised_text.replace(originals[i], x['revision'])
                        else:
                            revised_text = revised_text.replace(originals[i], '')
            diff = self.mark_differences(x['original'],x['revision'])
            x.update({'original_tagged':diff[0],'revision_tagged':diff[1],'combined_tagged':diff[2]})
            result2.append(x)

        original_analysis = self.analyser.analyze_cefr(text, outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels'],v=2)
        revision_analysis = self.analyser.analyze_cefr(revised_text, outputs=['sentences','wordlists','vocabulary_stats','tense_count','tense_term_count','tense_stats','clause_count','clause_stats','final_levels'],v=2)

        diff = self.mark_differences(text,revised_text)
        return {'revised_text':revised_text,'revised_text_tagged':diff[1],'original_text_tagged':diff[0],'combined_text_tagged':diff[2],'revisions':result2,'original_analysis':original_analysis, 'revision_analysis':revision_analysis}
    
    def mark_differences(self, str1, str2):
        words1 = str1.replace('\n',' \n ').split(' ')
        words2 = str2.replace('\n',' \n ').split(' ')

        diff = list(difflib.ndiff(words1, words2))
        marked_str1 = []
        marked_str2 = []
        both = []

        for d in diff:
            word = d[2:]
            if d.startswith('?'):
                continue
            if d.startswith('-'):
                marked_str1.append(f'<s>{word}</s>')
                both.append(f'<s>{word}</s>')
            elif d.startswith('+'):
                marked_str2.append(f'<b>{word}</b>')
                both.append(f'<b>{word}</b>')
            else:
                marked_str1.append(word)
                marked_str2.append(word)
                both.append(word)

        return ' '.join(marked_str1).replace(' \n ','\n'),' '.join(marked_str2).replace(' \n ','\n'), ' '.join(both).replace(' \n ','\n')
    
    