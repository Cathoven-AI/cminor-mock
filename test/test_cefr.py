from cminor.analyzers import AdoTextAnalyzer

analyser = AdoTextAnalyzer()

text = '''It will let others like you and play with you often.'''

analyser.analyze_cefr(text)

print(analyser.cefr.result['tense_term_count'])