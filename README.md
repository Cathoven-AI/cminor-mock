A collection of computational linguistic functions ranging from finding degree of decodebility to cross sentence similarity.
Now it also includes all the analysers.

# Changelog
### Version 1.5.9
###### 29 August 2022

##### CefrAnalyzer.classify_tense
+ [Fix] Infinitives in compound sentences now won’t be recognised as imperatives mistakenly.

##### ReadabilityAnalyzer.start_analyze
+ [Feature] Added readability_consensus

### Version 1.5.8
###### 12 August 2022

##### CefrAnalyzer.classify_tense
+ [Enhancement] Imperatives will now be recognised.
+ [Fix] "verb+doing" won't be recognised as "be doing"+"inf." tense anymore.

##### CefrAnalyzer.get_aux
+ [Fix] "to" in the infinitives' form is now always preserved.

##### CefrAnalyzer.get_verb_form
+ [Fix] Non-aux "have" now has "do" as form

### Version 1.5.7
###### 10 August 2022

##### CefrAnalyzer.get_clause
+ [Fix] Noun clauses as subjects are now properly recognised (which was affected by the changes in version 1.5.3).

### Version 1.5.6
###### 09 August 2022
+ [Fix] Recognition of compound sentences will now exclude clauses without a subject properly.

##### CefrAnalyzer.process
+ [Change] Level by length formula: ($$1.1^{length}-1$$) → ($$1.1^{length}-1.5$$), which is now capped at 7.
+ [Change] Final sentence CEFR level: the higher one of the level by length and the level by vocabulary → the average between the two.
+ [Change] CEFR clause of text is now taken at the 90th percentile of all sentences instead of the 80th.

### Version 1.5.5
###### 03 August 2022
+ [Enhancement] Clause level is now rounded to one decimal place.
+ [Fix] CEFR word count will now exclude SPACE.

##### CefrAnalyzerv.get_clause
+ [Feature] Recognition of compound sentences and subordinate clauses with conjunctions (and, or, so, but, etc.).

##### SolarSpacy.prevent_sbdv
+ [Fix] Sentences now won’t break in front of commas.

### Version 1.5.4
###### 03 August 2022

##### CefrAnalyzer.process
+ [Fix] Clause level of individual sentences is now shown properly.

### Version 1.5.3
###### 01 August 2022

##### CefrAnalyzer.clause2level
+ [Change] Level of “because” clauses: A2→A1

##### CefrAnalyzer.process
+ [Change] Clause level of individual sentences is now the higher value of the level by grammar (the highest one in that sentence) and the level by length ($$1.1^{length}-1$$). Clause level of text is taken at the 80th percentile of all sentences.

##### CefrAnalyzerv.get_clause
+ [Enhancement] "have got" which indicates possession is now tagged as A1 instead of A2.
+ [Fix] "let/make/etc. + do" is not classified as a clause anymore
+ [Fix] Noun clauses before their main clauses now will not be recognised as noun clauses.

##### SolarSpacy.prevent_sbdv
+ [Enhancement] Sentences now will not break at the end of a quote if the quote ends with a comma. Otherwise, if the quote is not followed by an uncapitalised letter, sentences will always break at the end of the quote.
+ [Fix] New line will now break sentences.
