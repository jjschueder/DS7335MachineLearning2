# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:23:31 2021

@author: jjschued
"""

# ==============================================================================
# File: ML2 HW1.py
# Project: Machine Learning 2 Homework 1 List Comprehension
# Author: Joe Schueder
# File Created: Jan 6, 2021
# ==============================================================================

from collections import Counter
import numpy as np

# ts# please fill in an explanation of each function and an example of how to use it below.

# List:
# append()
# extend()
# index()
# index(value, integer)
# insert()position
# remove()

# pop()
# pop(0)
# count()
# reverse()
# sort()
# [1]+[1]==1
# [y*2 for x in [[1
# [2]*[2]
# [1,2][1:]
# [x for x in [2,3]]
# [x for x in [1,2] if x ,2],[3,4]] for y in x]

# Tuple:
# 	count()
# 	index()
# about:blank#blocked
# build a dictionary from tuples
# unpack tuples

# Dicts:
# a_dict = {'I hate':'you', 'You should':’leave’}
# values()
# keys()
# items()
# has_key()
# ‘never’ in a_dict
# del a_dict['me']
# a_dict.clear()

# Ok enough by me do the rest on your own!
# use dir() to get built in functions
# Sets:
# 	Fill in yourself
# Strings:
# 	Fill in yourself
from collections import Counter
# 	Fill in yourself


# Bonus:
from itertools import *



flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

# 1. Build a counter object and use counter and confirm they have the same values.

cnt = Counter()

for word in flower_orders:
     cnt[word] += 1

cnt


# 2. Count how many objects have color W in them.

#first convert from list of strings to list of lists
mylist = []
listoflists = []
for word in flower_orders:
    mylist = list(word.split("/")) 
    listoflists.append(mylist)
    
#examine new list of lists to get counts
wcounter = 0
for word in listoflists:
    wcounter += word.count('W')    
    
# 3. Make histogram of colors
#create the counter object by merging into one big list then counting
import itertools
biglist = (list(itertools.chain.from_iterable(listoflists))) 
colorcounter = Counter()

for color in biglist:
    print(color)
    colorcounter[color] += 1

colorcounter
colorcountdict = dict(colorcounter)
#visualize with matplotlib
import matplotlib.pyplot as plt
plt.bar(colorcountdict.keys(), colorcountdict.values())
plt.show()

# 4. Rank the pairs of color pairs regardless of how many colors in order.

from itertools import permutations 
from itertools import combinations

paircombos = []
pairpermus = []
for i in range(len(listoflists)):
    print(i)
    combos = combinations(listoflists[i], 2)
    permus = permutations(listoflists[i], 2)
    paircombos.append(list(combos))
    pairpermus.append(list(permus))

bigpaircombolist = (list(itertools.chain.from_iterable(paircombos))) 
bigpairpermuslist = (list(itertools.chain.from_iterable(pairpermus))) 

paircounter = Counter()
for pair in bigpaircombolist:
    paircounter[pair] += 1
pairdict = dict(paircounter )
# 5. Rank the triplets of color pairs regardless of how many colors in order.

from itertools import permutations 
from itertools import combinations

tricombos = []
tripermus = []
for i in range(len(listoflists)):
    print(i)
    combos = combinations(listoflists[i], 3)
    permus = permutations(listoflists[i], 3)
    tricombos.append(list(combos))
    tripermus.append(list(permus))

bigtricombolist = (list(itertools.chain.from_iterable(tricombos))) 
bigtripermuslist = (list(itertools.chain.from_iterable(tripermus))) 

tricounter = Counter()
for pair in bigtricombolist:
    tricounter[pair] += 1
tridict = dict(tricounter)

# 6. Make dictionary of where keys are a color and values are what colors go with it

colorcountdict = dict(colorcounter)
colorcountdict
# 7. Make a graph showing the probability of having an edge between two colors based on 
#how often they co-occur.  (a numpy square matrix)
paircounter = Counter()
for pair in bigpaircombolist:
    paircounter[pair] += 1

denominator= sum(pairdict.values())

probdict = {'color':'prob'}
for k, v in pairdict.items():
    print ('k:',k)
    print ('v', v)
    percent = v / denominator * 100
    print(percent)
    d = {k:percent}
    probdict.update(d)

pairlist = list(paircounter)


#pairdict = dict(paircounter )
#pairarray = np.array(pairdict)
#npsquare = np.square(pairarray)
# 8. Make 10 business questions related to the questions we asked above.



dead_men_tell_taies = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']


# 1. Join everything
# initialize an empty string 
dmstring  = " " 
dmstring = dmstring.join(dead_men_tell_taies)

# 2. Remove spaces
dmstring = dmstring.replace(" ", "")


# 3. Occurrence probabilities for letters
denominator = len(dmstring)
import string
import collections
letter_counts = collections.Counter(dmstring)
letter_counts
letter_count_dict = dict(letter_counts)

#probdictdead = {'letter':'prob'}
probdictdead = {}
for k, v in letter_count_dict.items():
    print ('k:',k)
    print ('v', v)
    percent = v / denominator * 100
    print(percent)
    d = {k:percent}
    probdictdead.update(d)

# 4. Tell me transition probabilities for every letter pairs
trans = dict(Counter(zip(dmstring[:-1], dmstring[1:])))
denominator = sum(trans.values())


probtransdict = {}
for k, v in trans.items():
    print ('k:',k)
    print ('v', v)
    percent = v / denominator * 100
    print(percent)
    d = {k:percent}
    probtransdict.update(d)
# 5. Make a 26x26 graph of 4. in numpy
# #optional
# 6. plot graph of transition probabilities from letter to letter
#visualize with matplotlib
import matplotlib.pylab as plt

lists = sorted(probtransdict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
xlist = list(x)
reslist = [''.join(i) for i in xlist]
keys = list(probtransdict.keys()))
plt.bar(reslist, probtransdict.values())
plt.show()

# Unrelated:
# 7. Flatten a nested list
# Cool intro python resources:
# https://thomas-cokelaer.info/tutorials/python/index.html
