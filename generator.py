import openai, json, ast, warnings

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

        n_why = n//2
        json_format = '''[{"question": "Why is this the case?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 0},{"question": "What is this?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 2}]'''
        content = f'''Generate {n} multiple choice questions for this text. {n-n_why} of the questions are about main ideas and details. The other {n_why} are "why" questions. Each question has four choices. The answers must be logically correct and the other choices must be incorrect. You MUST arrange the questions as a Python list of dictionaries like this: {json_format}. Each dictionary is one question with keys "question", "choices", and "answer_index". The answer_index ranges from 0 to 3. I will need to parse the list directly using ast.literal_eval in Python.
        Text: {text}'''

        messages = [{"role": "system", "content": '''You are an English teacher who creates reading comprehension questions for an exam in Python code.'''},{"role": "user", "content": content}]
        if override_messages is None:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
            except Exception as e:
                return {'error':e.__class__.__name__,'detail':str(e)}
        else:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = override_messages
                )
            except Exception as e:
                return {'error':e.__class__.__name__,'detail':str(e)}
            
        response = completion['choices'][0]['message']['content'].strip()
        questions = self.parse_questions(response)

        if questions is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1, override_messages=messages+[{"role": completion['choices'][0]['message']['role'], "content": completion['choices'][0]['message']['content']},
                                                                                                              {"role": "user", "content": f"The questions you returned are not in Python dictionary format. Return them in as a Python list of dictionaries like this example: {json_format}"}])
                else:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1)
            else:
                return {'error':"SyntaxError",'detail':f"The bot didn't return the questions in Python dictionary format. response: {response}"}

        return questions
    
    def parse_questions(self, response):
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