import numpy as np
import openai, json, ast, warnings, os, sys, re

class AdoQuestionGenerator(object):
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key

    def generate_questions(self, text, n=10, kind='multiple_choice', auto_retry=3, override_messages=None):

        auto_retry = min(int(auto_retry),3)

        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key

        if kind=='multiple_choice':
            json_format = '''[{"question": "Why is this the case?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 0},{"question": "What is this?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 2}]'''

            content = f'''Your task is to generate high-order thinking multiple choice questions for a text. Each question has only one correct choice and three unambiguously incorrect choices.
            Follow the steps:
            1. Generate a high-order thinking question with 4 choices. One choice is logically correct and the other three are unambiguously incorrect.
            2. Verify each choice with the text to see if it can be the answer to the question.
            3. If more than one choices are a possible answer to the question, discard this question and start from the beginning.
            4. Repeat this process until you have {n} different questions.

            After you generate {n} questions, arrange them as a Python list of dictionaries in this format:
            ```{json_format}```

            Each dictionary must meet the following requirements:
            1. Each dictionary is one question with keys "question", "choices", and "answer_index".
            2. The answer_index ranges from 0 to 3.
            3. It can be parsed using ast.literal_eval in Python.


            Text:
            ```{text}```
            '''
        elif kind=='essay_question':
            json_format = '''[{"question": "Why is this the case?","answer": "Some answer"},{"question": "What is this?","answer": "Some answer"}]'''

            content = f'''Your task is to generate high-order thinking short answer questions for a text. Each question has only one correct answer.

            Answer rules:
            1. Use short phrases when possible. For example, the answer to "What are the freshwater forms of algae called?" should be "Charophyta." instead of "The freshwater forms of algae are called Charophyta."
            2. The answer should have a full stop at the end. For example, "Charophyta." instead of "Charophyta".
            
            Follow the steps:
            1. Generate a high-order thinking question.
            2. Answer the question yourself following the answer rules.
            3. Verify the answer in the text again.
            4. If the answer is not in the text, or if the answer is contradictory or ambiguous, discard this question and start from the beginning.
            5. Repeat this process until you have {n} different questions.

            After you generate {n} questions, arrange them as a Python list of dictionaries with keys "question" and "answer", like this:
            ```{json_format}```

            Each dictionary must meet the following requirements:
            1. Each dictionary is one question with keys "question" and "answer".
            2. It can be parsed using ast.literal_eval in Python.


            Text:
            ```{text}```
            '''

        messages = [{"role": "user", "content": content}]
        
        if override_messages is None:
            messages_to_send = messages
        else:
            messages_to_send = override_messages

        n_self_try = 3
        while n_self_try>0:
            try:
                completion = openai.ChatCompletion.create(
                    model='gpt-4-1106-preview',#"gpt-3.5-turbo",
                    messages=messages_to_send
                )
                break
            except Exception as e:
                n_self_try -= 1
                if n_self_try==0:
                    return {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

        response = completion['choices'][0]['message']['content'].strip(' `')
        questions = self.parse_questions(response)

        if questions is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1, override_messages=messages+[{"role": completion['choices'][0]['message']['role'], "content": completion['choices'][0]['message']['content']},
                                                                                                              {"role": "user", "content": f"The questions you returned are not in Python list of dictionary format. Return them as a Python list of dictionaries like this example: {json_format}"}])
                else:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1)
            else:
                return {'error':"SyntaxError",'detail':f"The bot didn't return the questions in Python dictionary format. Response: {response}"}

        if kind=='essay_question':
            for i in range(len(questions)):
                questions[i]['answer'] = questions[i]['answer'].capitalize()

        return questions
    
    def parse_questions(self, response):
        print(response)
        try:
            response = response[response.index('['):response.rfind(']')+1]
            questions = ast.literal_eval(response)
        except:
            try:
                response = response[response.index('['):response.rfind(']')+1]
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

    def create_text(self,level,n_words=300,topic=None,keywords=None,grammar=None,genre=None,ignore_keywords=True,
                    propn_as_lowest=True,intj_as_lowest=True,keep_min=True,
                    return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                    return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                    return_clause_stats=True,return_phrase_count=True,return_final_levels=True,return_modified_final_levels=True):
        assert type(grammar)==list or grammar is None, "grammar must be a list of strings."
        assert type(keywords)==list or keywords is None, "keywords must be a list of strings."
        
        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key
        prompt = self.construct_prompt(level=level,n_words=n_words,topic=topic,keywords=keywords,grammar=grammar,genre=genre)

        custom_dictionary = {}
        if keywords and ignore_keywords:
            for x in keywords:
                if type(x)==str:
                    custom_dictionary[x] = -1
                else:
                    custom_dictionary[tuple(x)] = -1
        if grammar:
            model = 'gpt-4'
        else:
            model = 'gpt-3.5-turbo'
        return self.execute_prompt(prompt,level,temp_results=[],model=model,
                                   propn_as_lowest=propn_as_lowest,intj_as_lowest=intj_as_lowest,keep_min=keep_min,custom_dictionary=custom_dictionary,
                                   return_sentences=return_sentences, return_wordlists=return_wordlists,return_vocabulary_stats=return_vocabulary_stats,
                                   return_tense_count=return_tense_count,return_tense_term_count=return_tense_term_count,return_tense_stats=return_tense_stats,
                                   return_clause_count=return_clause_count,return_clause_stats=return_clause_stats,return_phrase_count=return_phrase_count,
                                   return_final_levels=return_final_levels,return_modified_final_levels=return_modified_final_levels)

    def construct_prompt(self, level,n_words=300,topic=None,keywords=None,grammar=None,genre=None):
        int2cefr = {0:'A1',1:'A2',2:'B1',3:'B2',4:'C1',5:'C2'}
        target_level = int2cefr[level]
        max_length = int(round(np.log(level+0.5+1.5)/np.log(1.1),0))
        requirements = ['There should not be a title.']

        if topic:
            requirements.append(f"The topic is {topic}.")
        if genre:
            requirements.append(f"The genre (or type of text) is {genre}.")
        if keywords:
            keywords_to_join = []
            for x in keywords:
                if type(x)==str:
                    keywords_to_join.append(x)
                else:
                    keywords_to_join.append(f'{x[0]} ({x[1]})')
            requirements.append(f"It should include these words: {', '.join(keywords_to_join)}.")
        if grammar:
            grammar_list = ''
            for x in grammar:
                grammar_list += '\t'+x
                if x.lower() in grammar_list:
                    grammar_list += ': '+self.grammar_description[x.lower()]+';\n'
            requirements.append("Use these grammar structures many times:\n"+grammar_list)
        
        requirements.append(f"It should be around {n_words} words.")
        #requirements.append('''Don't use style text.''')
        requirements.append('''Arrange the result in json format like this:\n```{"text": the text}```\nIt should be parsed directly. Don't include other notes, tags or comments.''')
        
        if level<=2:
            prompt = f'''
You task is to use simple language to write a text at CEFR {target_level} level for elementary English learners. The difficulty of vacubalary is important. You must choose your words carefully and use only simple words. Don't use technical or academic vocabulary.

{target_level} level texts should meet these requirements:
1. Each sentence is not longer than {max_length} words.
2. The vocabulary should be simple and below CEFR {target_level} level.

In the meantime, the text should meet the following requirements:
'''
        else:
            prompt = f'''
You task is to write a {genre} text at CEFR {target_level} level.

{target_level} level texts should meet these requirements:
1. Each sentence is not longer than {max_length} words.
2. The vocabulary should not be more difficult than CEFR {target_level} level.

In the meantime, the text should meet the following requirements:
'''

        for i, x in enumerate(requirements):
            prompt += f"{i+1}. {x}\n"
        
        return prompt.strip('\n').replace(' None ',' ')

    def execute_prompt(self,prompt,level,auto_retry=3,temp_results=[],model='gpt-3.5-turbo',
                       propn_as_lowest=True,intj_as_lowest=True,keep_min=True,custom_dictionary={},
                      return_sentences=True, return_wordlists=True,return_vocabulary_stats=True,
                      return_tense_count=True,return_tense_term_count=True,return_tense_stats=True,return_clause_count=True,
                      return_clause_stats=True,return_phrase_count=True,return_final_levels=True,return_modified_final_levels=True):
        model = 'gpt-4-1106-preview'
        n_trials = len(temp_results)+1
        print(f"Trying {n_trials}")
        # if n_trials>=3:
        #     model = "gpt-4"
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=1
        )
        print(completion['choices'][0]['message']['content'])
        text = self.parse_response(completion['choices'][0]['message']['content'])
        result = self.analyser.analyze_cefr(text,propn_as_lowest=propn_as_lowest,intj_as_lowest=intj_as_lowest,keep_min=keep_min,custom_dictionary=custom_dictionary,
                        return_sentences=return_sentences, return_wordlists=return_wordlists,return_vocabulary_stats=return_vocabulary_stats,
                        return_tense_count=return_tense_count,return_tense_term_count=return_tense_term_count,return_tense_stats=return_tense_stats,return_clause_count=return_clause_count,
                        return_clause_stats=return_clause_stats,return_phrase_count=return_phrase_count,
                        return_final_levels=return_final_levels,return_modified_final_levels=return_modified_final_levels,return_result=True)
        if int(result['final_levels']['general_level'])!=level:
            if auto_retry>0:
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

    def parse_response(self, response):
        response = response[response.index('{'):response.rfind('}')+1]
        text = json.loads(response, strict=False)['text']
        try:
            response = response[response.index('{'):response.rfind('}')+1]
            text = json.loads(response, strict=False)['text']
        except:
            try:
                response = response[response.index('{'):response.rfind('}')+1]
                text = ast.literal_eval(response)['text']
            except:
                return None
        return text