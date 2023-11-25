import Levenshtein as lev
import numpy as np

def baidu_cefr_decorator(func):
    def inner(*args, **kwargs):
        params = {'return_sentences':False, 'return_wordlists':False,'return_vocabulary_stats':False,
                    'return_tense_count':False,'return_tense_term_count':False,'return_tense_stats':False,'return_clause_count':False,
                    'return_clause_stats':False,'return_phrase_count':False,'return_final_levels':True}
        kwargs.update(params)
        result = func(*args, **kwargs)

        if len(result['exam_stats']['exam_grades'])==0:
            exam_grades = '无'
        else:
            exam_grades = ' / '.join(result['exam_stats']['exam_grades'])

        message = '''以下是文章的CEFR难度分析结果
整体难度：{}
词汇难度：{}
时态难度：{}
句法难度：{}

对应剑桥英语考试体系中的以下能力水平：
· 剑桥英语量表分数（Cambridge Scale Score）：{}
· 剑桥英语考试等级：{}
· 雅思：{}
'''.format(
    level_str_to_cn(result['final_levels_str']['general_level']),
    level_str_to_cn(result['final_levels_str']['vocabulary_level']),
    level_str_to_cn(result['final_levels_str']['tense_level']),
    level_str_to_cn(result['final_levels_str']['clause_level']),
    result['exam_stats']['cambridge_scale_score'],
    exam_grades,
    result['exam_stats']['ielts']
    )
        return {'message':message+'''\n\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}
    return inner


def baidu_catile_decorator(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)

        catile_score = round(result['scores']['catile'] / 10) * 10

        age_range = result['age']
        age = age_range[0].split('-')[0] + '-' + age_range[-1].split('-')[1]

        grade_range = result['grade']
        grade = str(grade_range[0]) + '-' + str(grade_range[-1])

        difficult_words = ', '.join(result['difficult_words'])
        longest_sentence = result['longest_sentence']
        mean_sent_length = result['mean_sent_length']

        message = '''以下是文章的Catile难度分析结果
Catile值：{}
年龄：{}岁
年级（美国体系）：{}年级
难词：{}
最长句子：{}
平均句长：{}个单词
'''.format(catile_score, age, grade, difficult_words, longest_sentence, mean_sent_length)
        return {'message':message+'''\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}
    return inner


def baidu_adaptor_decorator(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)

        after_str = result['after_str']
        after_exam_stats = result['after_exam_stats']

        target_level = kwargs.get('target_level',7)
        if type(target_level)==str:
            target_level = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}[target_level.upper()]
        ignored_words = []

        alert = ''
        if target_level < int(result['before']['general_level']) and int(result['after']['general_level']) > target_level:
            alert = 'difficult'
            if result.get('modified_after_levels'):
                final_levels = result['modified_after_levels'].get('final_levels')
                final_levels_str = result['modified_after_levels'].get('final_levels_str')
                exam_stats = result['modified_after_levels'].get('exam_stats')
                if final_levels and int(final_levels['general_level']) <= target_level:
                    for x in result['modified_after_levels']['ignored_words']:
                        ignored_words.append(f'"{x.split("_")[0]}"')
                    alert = 'modified'

        message = ''
        if alert == 'difficult':
            message = '这篇文章的主题对所选级别难度偏大，改写后可能仍会偏难。'
        elif alert == 'modified':
            if len(ignored_words) > 1:
                message = ', '.join(ignored_words) + '是重要的关键词，不可替换。改写后文章的难度计算已忽略这些单词，建议在阅读前先学习这些单词。'
            else:
                message = ignored_words[0] + '是重要的关键词，不可替换。改写后文章的难度计算已忽略这个单词，建议在阅读前先学习它。'
            after_str = final_levels_str
            after_exam_stats = exam_stats

        if len(result['before_exam_stats']['exam_grades'])==0:
            before_exam_grades = '无'
        else:
            before_exam_grades = ' / '.join(result['before_exam_stats']['exam_grades'])
        if len(after_exam_stats['exam_grades'])==0:
            after_exam_grades = '无'
        else:
            after_exam_grades = ' / '.join(after_exam_stats['exam_grades'])


        message0 = '''以下是改写后的文章和难度分析结果：
{}

{}

原文难度：
· 整体难度：{}
· 词汇难度：{}
· 时态难度：{}
· 句法难度：{}
原文对应剑桥英语考试体系中的以下能力水平：
· 剑桥英语量表分数（Cambridge Scale Score）：{}
· 剑桥英语考试等级：{}
· 雅思：{}

改写后难度：
· 整体难度：{}
· 词汇难度：{}
· 时态难度：{}
· 句法难度：{}
改写后对应剑桥英语考试体系中的以下能力水平：
· 剑桥英语量表分数（Cambridge Scale Score）：{}
· 剑桥英语考试等级：{}
· 雅思：{}
'''.format(result['adaptation'],
           message,
           level_str_to_cn(result['before_str']['general_level']),
           level_str_to_cn(result['before_str']['vocabulary_level']),
           level_str_to_cn(result['before_str']['tense_level']),
           level_str_to_cn(result['before_str']['clause_level']),
           result['before_exam_stats']['cambridge_scale_score'],
           before_exam_grades,
           result['before_exam_stats']['ielts'],
           level_str_to_cn(after_str['general_level']),
           level_str_to_cn(after_str['vocabulary_level']),
           level_str_to_cn(after_str['tense_level']),
           level_str_to_cn(after_str['clause_level']),
           after_exam_stats['cambridge_scale_score'],
           after_exam_grades,
           after_exam_stats['ielts'])

        return {'message':message0+'''\n\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}
    return inner


def baidu_text_generator_decorator(func):
    def inner(*args, **kwargs):
        params = {'return_sentences':False, 'return_wordlists':False,'return_vocabulary_stats':False,
                    'return_tense_count':False,'return_tense_term_count':False,'return_tense_stats':False,'return_clause_count':False,
                    'return_clause_stats':False,'return_phrase_count':False,'return_final_levels':True}
        kwargs.update(params)
        result = func(*args, **kwargs)

        if len(result['result']['exam_stats']['exam_grades'])==0:
            exam_grades = '无'
        else:
            exam_grades = ' / '.join(result['result']['exam_stats']['exam_grades'])

        message = '''以下是根据条件生成的文章及难度信息
{}

整体难度：{}
词汇难度：{}
时态难度：{}
句法难度：{}

对应剑桥英语考试体系中的以下能力水平：
· 剑桥英语量表分数（Cambridge Scale Score）：{}
· 剑桥英语考试等级：{}
· 雅思：{}'''.format(result['text'],
                  level_str_to_cn(result['result']['final_levels_str']['general_level']),
                  level_str_to_cn(result['result']['final_levels_str']['vocabulary_level']),
                  level_str_to_cn(result['result']['final_levels_str']['tense_level']),
                  level_str_to_cn(result['result']['final_levels_str']['clause_level']),
                  result['result']['exam_stats']['cambridge_scale_score'],
                  exam_grades,
                  result['result']['exam_stats']['ielts'])
        return {'message':message+'''\n\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}
    return inner


def baidu_question_generator_decorator(func):
    def inner(*args, **kwargs):
        kind = kwargs.get('kind','multiple_choice')
        kinds_cn = {'multiple choice':'multiple_choice',
                    '选择题':'multiple_choice',
                    '单选题':'multiple_choice',
                    '单项选择题':'multiple_choice',
                    '单选':'multiple_choice',
                    'multiple choice cloze':'multiple_choice_cloze',
                    '选择填空题':'multiple_choice_cloze',
                    '选择填空':'multiple_choice_cloze',
                    '完形填空题':'multiple_choice_cloze',
                    '完形填空':'multiple_choice_cloze',
                    '完形':'multiple_choice_cloze',
                    '填词':'sentence_completion',
                    '填词题':'sentence_completion',
                    '填空':'sentence_completion',
                    '填空题':'sentence_completion',
                    '句子填空题':'sentence_completion',
                    '判断题':'true_false',
                    '判断':'true_false',
                    '判断对错':'true_false',
                    'true or false':'true_false',
                    '对错':'true_false',
                    'true false not given':'true_false_not_given',
                    '简答题':'short_answer',
                    '简答':'short_answer',
                    '问答题':'short_answer',
                    '问答':'short_answer'}
        if kind in kinds_cn:
            kind = kinds_cn[kind]
        else:
            keys = list(kinds_cn.keys())
            k = np.argmin([lev.distance(kind, x) for x in keys])
            kind = kinds_cn[keys[k]]

        params = {'explanation':True,'explanation_language':'simplified Chinese','question_language':'English','answer_position':True,'kind':kind}
        kwargs.update(params)
        if 'n' not in kwargs:
            kwargs['n'] = 5
        result = func(*args, **kwargs)

        if kind == 'multiple_choice':
            letters = ['A','B','C','D']
            answers = ''
            questions = ''
            for i, question in enumerate(result):
                questions += f'{i+1}. {question["question"]}\n'
                for j, choice in enumerate(question['choices']):
                    questions += f'{letters[j]}. {choice}\n'
                questions += '\n'
                answers += f'{i+1}. {letters[question["answer_index"]]}\n解析：{question.get("explanation","")}（{question.get("answer_position","")}）\n\n'
            message = f'''以下是生成的单选题\n\n{questions}\n答案：\n\n{answers}'''
        elif kind == 'short_answer':
            answers = ''
            questions = ''
            for i, question in enumerate(result):
                questions += f'{i+1}. {question["question"]}\n\n'
                answers += f'{i+1}. {question["answer"]}\n解析：{question.get("explanation","")}（{question.get("answer_position","")}）\n\n'
            message = f'''以下是生成的简答题\n\n{questions}\n答案：\n\n{answers}'''
        elif kind == 'true_false':
            answers = ''
            questions = ''
            for i, question in enumerate(result):
                questions += f'{i+1}. {question["question"]}\n\n'
                answers += f'{i+1}. {question["answer"]}\n解析：{question.get("explanation","")}（{question.get("answer_position","")}）\n\n'
            message = f'''以下是生成的判断题(True or False)\n\n{questions}\n答案：\n\n{answers}'''
        elif kind == 'true_false_not_given':
            answers = ''
            questions = ''
            for i, question in enumerate(result):
                questions += f'{i+1}. {question["question"]}\n\n'
                answers += f'{i+1}. {question["answer"]}\n解析：{question.get("explanation","")}（{question.get("answer_position","")}）\n\n'
            message = f'''以下是生成的判断题(True/False/Not Given)\n\n{questions}\n答案：\n\n{answers}'''
        elif kind == 'sentence_completion':
            words = []
            answers = ''
            questions = ''
            for i, question in enumerate(result):
                questions += f'{i+1}. {question["sentence"]}\n'
                words.append(question['answer'])
            index = np.arange(len(words))
            np.random.shuffle(index)
            for i, j in enumerate(index):
                answers += f'{i+1}. {words[j]}\n'
            words = ', '.join(words)
            message = f'''以下是生成的填空题\n\n请用以下单词填空：{words}\n\n{questions}\n答案：\n\n{answers}'''
        elif kind == 'multiple_choice_cloze':
            letters = ['A','B','C','D']
            answers = ''
            questions = ''
            for i, question in enumerate(result['questions']):
                questions += f'{i+1}. '
                for j, choice in enumerate(question['choices']):
                    questions += f'{letters[j]}. {choice}    '
                questions += '\n'
                answers += f'{i+1}. {letters[question["answer_index"]]}\n解析：{question.get("explanation","")}\n\n'
            message = f'''以下是生成的完形填空练习\n\n{result['text']}\n\n{questions}\n答案：\n\n{answers}'''
        else:
            message = '暂不支持该类型题目的生成。支持的题型有：单选题，完形填空题，判断题，简答题，句子填空题。'

        return {'message':message+'''\n\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}

    return inner


def level_str_to_cn(s):
    parts = s.split('.')
    if len(parts) == 1:
        return '母语级别'
    else:
        if int(parts[1]) <= 3:
            stage = "初阶"
        elif int(parts[1]) >= 7:
            stage = "高阶"
        else:
            stage = "中阶"
        return f'{s}（{parts[0]}{stage}）'