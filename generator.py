import openai, json, ast

def generate_questions(text, n=10, auto_retry=True):
    n_why = n//2

    json_format = '''[{"question": "Why is this the case?","choices": ["Some choice","Some choice","Some choice","Some choice"],"answer_index": 0}]'''

    content = f'''Generate ten multiple choice questions for this text. {n-n_why} of them are questions about main ideas and details. The other {n_why} are "why" questions. Each question has four choices. The answers must be logically correct and the other choices must be incorrect. Return the results as "question", "choices", and "answer_index" in json format like this: {json_format}. The answer_index ranges from 0 to 3.

    Text: {text}'''
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": '''You are an English teacher creating reading comprehension questions for an exam.'''},{"role": "user", "content": content}]
    )
    try:
        questions = json.loads(completion['choices'][0]['message']['content'].strip())
    except:
        try:
            questions = ast.literal_eval(completion['choices'][0]['message']['content'].strip())
        except Exception as e:
            if auto_retry:
                return generate_questions(text, n=n, auto_retry=False)
            else:
                raise e
    return questions