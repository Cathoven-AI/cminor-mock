# Spacy stuff
import spacy, re
from spacy.language import Language
from .. import modify_text
standardize_old = modify_text.standardize_old


class SolarSpacy(Language):
	def __init__(self, spacy_pipeline):
		self.nlp_no_sbd = spacy.load(spacy_pipeline)
		self.nlp_no_sbd.tokenizer = SolarSpacy.custom_tokenizer(self.nlp_no_sbd)
		#self.nlp_no_sbd.add_pipe('set_custom_boundaries', before='parser')
		self.nlp_no_sbd.add_pipe('prevent_sbd', before='parser')

		# initialize all other spacy.language components
		self.vocab = self.nlp_no_sbd.vocab
		self.meta = self.nlp_no_sbd.meta
		self._components = self.nlp_no_sbd._components
		self._disabled = self.nlp_no_sbd._disabled

	def __call__(self, text):
		return self.get_doc(text)

	def get_doc(self, text):
		#text = standardize(text)
		text = standardize_old(text)
		return self.nlp_no_sbd(text)

	def explain(self,tag):
		return spacy.explain(tag)

	# @spacy.Language.component('set_custom_boundaries')
	# def set_custom_boundaries(doc):
	#	 for token in doc:
	#		 if token.i<len(doc)-1 and ("\n" in token.orth_ or token.orth_ in ["?","!","."] and ((bool(token.whitespace_)==True) or doc[token.i+1].orth_ in ['[','(','{'])):
	#			 doc[token.i+1].is_sent_start = True
	#			 if token.i<len(doc)-2:
	#				 doc[token.i+2].is_sent_start = False
	#	 return doc


	# @spacy.Language.component('sent_start_check')
	# def sent_start_check(doc):
	# 	for token in doc:
	#		 if token.is_sent_start and token.orth_ in [",","!","."]:
		
	#	 for token in doc:
	#		 if token.i<len(doc)-1 and ("\n" in token.orth_ or token.orth_ in ["?","!","."] and ((bool(token.whitespace_)==True) or doc[token.i+1].orth_ in ['[','(','{'])):
	#			 doc[token.i+1].is_sent_start = True
	#			 if token.i<len(doc)-2:
	#				 doc[token.i+2].is_sent_start = False
	#	 return doc

	@Language.component('prevent_sbd')
	def prevent_sbd(doc):
		""" Ensure that SBD does not run on tokens inside quotation marks and brackets. """
		quote_open = False
		bracket_open = False
		can_sbd = True
		sent_end = False
		counter = 0
		
		for token in doc:
			if not can_sbd:
				counter += 1
				if sent_end:
					if not token.is_punct:
						token.is_sent_start = True
					elif counter > 1 and token.text != '"':
							token.is_sent_start = False
					sent_end = False
				elif counter > 1 and token.text != '"':
						token.is_sent_start = False
				if token.is_punct and token.text.strip() not in [',', ';', '"', '—', '-', "'", '(', ')', '[', ']','']:
					sent_end = True
			else:
				counter = 0
				if token.text.strip()!='' and token.i>=1:
					j = token.i-1
					while (j>0 and doc[j].text.strip(' ')==''):
						j -= 1
					if doc[j].text == '"':
						if bool(re.match('[a-z,\)\]]',token.text[0])):
							token.is_sent_start = False
						else:
							k = j-1
							while (k>0 and doc[k].text.strip(' ')==''):
								k -= 1
							if k>=0 and doc[k].text.strip(' ')!=',':
								token.is_sent_start = True
					elif '\n' in doc[j].text:
						doc[j].is_sent_start = False
						token.is_sent_start = True
						if token.i<len(doc)-2:
							doc[token.i+1].is_sent_start = False
				
			# Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
			# Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct
			if token.text == '"':
				quote_open = not quote_open
			elif token.is_bracket and token.is_left_punct:
				bracket_open = True
			elif token.is_bracket and token.is_right_punct:
				bracket_open = False

			can_sbd = not (quote_open or bracket_open)

		return doc

	def custom_tokenizer(nlp):
		from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
		from spacy.tokenizer import Tokenizer
		inf = list(nlp.Defaults.infixes)			   # Default infixes
		inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")	# Remove the generic op between numbers or between a number and a -
		inf = tuple(inf)							   # Convert inf to tuple
		infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)", '''[?!&:,()”;"—\[\]]'''])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
		infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
		infix_re = compile_infix_regex(infixes)

		return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
									suffix_search=nlp.tokenizer.suffix_search,
									infix_finditer=infix_re.finditer,
									token_match=nlp.tokenizer.token_match,
									rules=nlp.Defaults.tokenizer_exceptions)