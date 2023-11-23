def baidu_cefr_decorator(func):
    def inner(*args, **kwargs):
        params = {'return_sentences':False, 'return_wordlists':False,'return_vocabulary_stats':False,
                    'return_tense_count':False,'return_tense_term_count':False,'return_tense_stats':False,'return_clause_count':False,
                    'return_clause_stats':False,'return_phrase_count':False,'return_final_levels':True}
        kwargs.update(params)
        result = func(*args, **kwargs)

        message = '''以下是文章的CEFR难度分析结果
整体难度：{}
词汇难度：{}
时态难度：{}
句法难度：{}
'''.format(level_str_to_cn(result['final_levels_str']['general_level']), level_str_to_cn(result['final_levels_str']['vocabulary_level']),
            level_str_to_cn(result['final_levels_str']['tense_level']), level_str_to_cn(result['final_levels_str']['clause_level']))
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
                if final_levels and int(final_levels['general_level']) <= target_level:
                    for x in result['modified_after_levels']['ignored_words']:
                        ignored_words.append(f'"{x.split("_")[0]}"')
                    alert = 'modified'

        message = ''
        if alert == 'difficult':
            message = '这篇文章的主题对所选级别难度偏大，改写后难度可能仍会偏难。'
        elif alert == 'modified':
            if len(ignored_words) > 1:
                message = ', '.join(ignored_words) + '是重要的关键词，不可替换。改写后文章的难度计算已忽略这些单词，建议在阅读前先学习这些单词。'
            else:
                message = ignored_words[0] + '是重要的关键词，不可替换。改写后文章的难度计算已忽略这个单词，建议在阅读前先学习它。'
            after_str = final_levels_str

        message0 = '''以下是改写后的文章和难度分析结果：
{}

{}

原文难度：
    整体难度：{}
    词汇难度：{}
    时态难度：{}
    句法难度：{}

改写后难度：
    整体难度：{}
    词汇难度：{}
    时态难度：{}
    句法难度：{}
'''.format(result['adaptation'],
           message,
           level_str_to_cn(result['before_str']['general_level']),
           level_str_to_cn(result['before_str']['vocabulary_level']),
           level_str_to_cn(result['before_str']['tense_level']),
           level_str_to_cn(result['before_str']['clause_level']),
           level_str_to_cn(after_str['general_level']),
           level_str_to_cn(after_str['vocabulary_level']),
           level_str_to_cn(after_str['tense_level']),
           level_str_to_cn(after_str['clause_level']))

        return {'message':message0+'''\n\n(访问www.cathoven.com获取更多辅助英语学习与教学的AI工具。)'''}
    return inner


def baidu_text_generator_decorator(func):
    def inner(*args, **kwargs):
        params = {'return_sentences':False, 'return_wordlists':False,'return_vocabulary_stats':False,
                    'return_tense_count':False,'return_tense_term_count':False,'return_tense_stats':False,'return_clause_count':False,
                    'return_clause_stats':False,'return_phrase_count':False,'return_final_levels':True}
        kwargs.update(params)
        result = func(*args, **kwargs)
        message = '''以下是根据条件生成的文章及难度信息
{}

整体难度：{}
词汇难度：{}
时态难度：{}
句法难度：{}'''.format(result['text'],
                  level_str_to_cn(result['result']['final_levels_str']['general_level']),
                  level_str_to_cn(result['result']['final_levels_str']['vocabulary_level']),
                  level_str_to_cn(result['result']['final_levels_str']['tense_level']),
                  level_str_to_cn(result['result']['final_levels_str']['clause_level']))
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
        return f'{s}({parts[0]}{stage})'