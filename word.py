#import pronouncing
from g2p_en import G2p
from . import math_functs
import numpy as np

# This package provides phonemes when a string is given
g2p = G2p()


#DECODABILITY
def decoding_degree(s):
    if type(s) == str:
        decode = decode_str(s)
        degree = decode_degree_dict.get(decode, 8)
        
        if any_dipthongs(s):
            if degree != 6:
                degree = 7
        
        return degree
    elif type(s) == list:
        decoding_degrees_arr = []
        for word in s:
            decoding_degrees_arr.append(decoding_degree(word))
        return decoding_degrees_arr
    else:
        raise Exception("The type of the input should either be a string or a list of strings")

def decode_str(s):
    s = s.lower()
    decoding = ''
    for l in s:
        if l in vowel:
            decoding += 'v'
        else:
            decoding += 'c'
            
    if len(s) >= 3:
        # words that end with 'e'
        if s[-1] == 'e':
            decoding = decoding[0:-1] + 'e'
            
        # words with 'r' as a second last char
        if s[-2] == 'r' and decoding[-3] == 'v':
            decoding = decoding[0:-2] + 'r' + decoding[-1]

        # words with 'l' as a second last char
        elif s[-2] == 'l' and decoding[-3] == 'v' and decoding[-1] == 'c':
            decoding = decoding[0:-2] + 'l' + decoding[-1]
            
    if len(s) >= 2:
        if s[-1] == 'y':
            decoding = decoding[0:-1] + 'v'
            
        # words that end with 'r'
        elif s[-1] == 'r':
            decoding = decoding[0:-1] + 'r'
        
        # words that end with 'll'
        if s[-2:] == 'll':
            decoding = decoding[0:-2] + 'll'

        # words that end with 'gh'
        elif s[-2:] == 'gh':
            decoding = decoding[0:-2] + 'gh'

    return decoding


dipthong_phones = ['AW', 'OY', 'AO']

def any_dipthongs(s):
    is_there = False

    phones = g2p(s.lower())
    for phone in phones:
        # remove numbers
        phone = ''.join([i for i in phone if not i.isdigit()])

        if phone in dipthong_phones:
            is_there = True
            break
    return is_there



vowel = ['a', 'e', 'i', 'o', 'u']

decode_degree_dict = {
    'vce': 4,
    'cvce': 4,
    'ccvce': 4,
    'cccvce': 4,
    'cvr': 6,
    'ccvr': 6,
    'vrc': 6,
    'cvrc': 6,
    'ccvrc': 6,
    'vll': 6,
    'cvll': 6,
    'ccvll': 6,
    'cvlc': 6,
    'ccvlc': 6,
    'cvvlc': 6,
    'ccvvlc': 6,
    'v': 1,
    'cv': 1,
    'cvc': 2,
    'vc': 2,
    'cce': 3,
    'ccv': 3,
    'vcc': 3,
    'vccc': 3,
    'ccvc': 3,
    'cccvc': 3,
    'cvcc': 3,
    'cvccc': 3,
    'ccvcc': 3,
    'ccvccc': 3,
    'cccvcc': 3,
    'cccvccc': 3,
    'cvv': 5,
    'cve': 5,
    'ccve': 5,
    'cvvc': 5,
    'cvvcc': 5,
    'ccvv': 5,
    'ccvvc': 5,
    'ccvvcc':5,
    'vvc': 5,
    'vvcc': 5
}


# Provide the original word forms not lemmas
def decoding_degree_mean(word_array):
    decoding_degrees_arr= decoding_degree(word_array)
    return np.array(decoding_degrees_arr).mean()

def decoding_degree_high_mean(word_array):
    decoding_degrees_arr= decoding_degree(word_array)
    return math_functs.high_mean(decoding_degrees_arr)

# Provide the original word forms not lemmas
def decoding_degree_stats(word_array):
    decoding_degrees_arr= decoding_degree(word_array)

    index_highest = np.argsort(decoding_degrees_arr)[-1]
    decoding_degree_word_highest = decoding_degrees_arr[index_highest]

    return {"decoding_degree_word_highest": decoding_degree_word_highest,
            "decoding_degree_mean": np.array(decoding_degrees_arr).mean(),
            "decoding_degree_high_mean": math_functs.high_mean(decoding_degrees_arr)}


# SYLLABLES

# This function takes a string or a list as an input.
def count_syllables(s):
    if type(s) == str:
        phonomes = g2p(s.lower())
        counter = 0
        for phonome in phonomes:
            if any(char.isdigit() for char in phonome):
                counter += 1
        return counter
    elif type(s) == list:
        n_syllables = []
        for word in s:
            n_syllables.append(count_syllables(word))
        return n_syllables
    else:
        raise Exception("The type of the input should either be a string or a list of strings")


# Provide the original word forms not lemmas
def n_syllables_mean(word_array):
    n_syllables = count_syllables(word_array)
    return np.array(n_syllables).mean()

def n_syllables_high_mean(word_array):
    n_syllables = count_syllables(word_array)
    return math_functs.high_mean(n_syllables)


def n_syllables_stats(word_array):
    n_syllables = count_syllables(word_array)
    index_highest = np.argsort(n_syllables)[-1]
    n_syllables_word_highest = n_syllables[index_highest]

    return {"n_syllables_total": np.sum(n_syllables),
            "n_syllables_mean": np.array(n_syllables).mean(),
            "n_syllables_high_mean": math_functs.high_mean(n_syllables),
            "n_syllables_word_highest": n_syllables_word_highest}