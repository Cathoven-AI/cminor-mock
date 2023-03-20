import openai, json, ast, warnings

class AdoQuestionGenerator(object):
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key

    def generate_questions(self, text, n=10, kind='multiple_choice', auto_retry=True):

        if self.openai_api_key is None:
            warnings.warn("OpenAI API key is not set. Please assign one to .openai_api_key before calling.")
            return None
        else:
            openai.api_key = self.openai_api_key

        n_why = n//2

        json_format = '''[{"question": "Why is this the case?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 0}]'''

        content = f'''Generate {n} multiple choice questions for this text. {n-n_why} of them are questions about main ideas and details. The other {n_why} are "why" questions. Each question has four choices. The answers must be logically correct and the other choices must be incorrect. Return the results as "question", "choices", and "answer_index" in json format like this: {json_format}. The answer_index ranges from 0 to 3.

        Text: {text}'''
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": '''You are an English teacher creating reading comprehension questions for an exam.'''},{"role": "user", "content": content}]
            )
        except Exception as e:
            return {'error':e.__class__.__name__,'detail':str(e)}
        
        try:
            questions = json.loads(completion['choices'][0]['message']['content'].strip())
        except:
            try:
                questions = ast.literal_eval(completion['choices'][0]['message']['content'].strip())
            except Exception as e:
                if auto_retry:
                    return self.generate_questions(text, n=n, kind=kind, auto_retry=False)
                else:
                    return {'error':e.__class__.__name__,'detail':str(e)}
        return questions