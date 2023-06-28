import openai, json, ast, warnings, os, sys

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
            Follow the steps:
            1. Generate a high-order thinking question with an answer.
            2. Verify the answer in the text.
            3. If the answer is not in the text, or if the answer is contradictory or ambiguous, discard this question and start from the beginning.
            4. Repeat this process until you have {n} different questions.

            After you generate {n} questions, arrange them as a Python list of dictionaries in this format:
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
                    model="gpt-3.5-turbo-0613",
                    messages=messages_to_send
                )
                break
            except Exception as e:
                n_self_try -= 1
                if n_self_try==0:
                    return {'error':e.__class__.__name__,'detail':f"(Tried 3 times.) "+str(e)}
                print(os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1],'line',sys.exc_info()[2].tb_lineno, e, "Retrying",3-n_self_try)

        response = completion['choices'][0]['message']['content'].strip()
        questions = self.parse_questions(response)

        if questions is None:
            if auto_retry>0:
                if auto_retry%2==1:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1, override_messages=messages+[{"role": completion['choices'][0]['message']['role'], "content": completion['choices'][0]['message']['content']},
                                                                                                              {"role": "user", "content": f"The questions you returned are not in Python dictionary format. Return them as a Python list of dictionaries like this example: {json_format}"}])
                else:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=auto_retry-1)
            else:
                return {'error':"SyntaxError",'detail':f"The bot didn't return the questions in Python dictionary format. Response: {response}"}

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