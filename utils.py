#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools


# In[2]:


"""
inverse mapping of tokenized text to list text
Input: doc_tokens (['B', '##O', '##O', '##K', ...])
Output: ['BOOK', ...]
"""
def inv_map(tokenized_text):
  text = []
  for i in tokenized_text:
    if len(i) < 3:
      text.append(i)
    elif i[0:2] != '##':
      text.append(i)
    else:
      text[-1] = text[-1] + i[2:]
  return text


# In[3]:


"""
Input: Document in list form, including punctuation as elements of list
Output: List of indexes of all sentence separating punctuation ('.' '?' '!')
e.g [1,30,33,40,...]
Can easily be adapted to include ','
"""

def punctuation_indices(document):
  indices = [i for i, x in enumerate(document) if x == '.' or x == '?' or x == '!']
  prenames = ['Dr', 'Mr', 'Mrs', 'Miss', 'Ms']
  false_indices = []
  for i in range(len(document)):
    if document[i] == '.':
      if document[i-1] in prenames:
        false_indices.append(i)
  removals = set(false_indices)
  indices = [x for x in indices if x not in removals]
  return indices


# In[4]:


"""
Input: list of sentence separating punctuation indices 
and clusters (entity id, and all its mentions' indices), e.g. [[(3,4), (130,133), 1],[(5,5), (399,399), 2],...]
Output: Augmented list of punctuation indices (includes the mention indices in between the punctuation indices,
and has the corresponding entity id in the 2nd column)
e.g. [[2,0],[3,1],[49,0],[87,0],[130,1],...]
"""
def add_entity_indices(punc_indices, clusters):
  augmented_list = punc_indices.copy()
  for i in range(len(augmented_list)):
    augmented_list[i] = [augmented_list[i],0]

  index_additives = np.zeros(len(punc_indices))

  for cluster in clusters:
    starting_index = 0
    for i in range(len(cluster)-1):
      for j in range(len(punc_indices[starting_index:])-1):
        if cluster[i][0] > punc_indices[starting_index + j] and cluster[i][0] < punc_indices[starting_index + j+1]:
          index_additives[starting_index + j:] += 1
          augmented_list.insert(starting_index + j + int(index_additives.item(starting_index + j)), [cluster[i][0],[cluster[-1], cluster[i]]])
          starting_index += j 
          break
  return np.asarray(augmented_list)


# In[5]:


"""
Input: Augmented list of punctuation indices with character mentions and characters 
(represented by number, e.g. Achilles = 1, agamemnon = 2)
Output: Shared sentence indices and corresponding character numbers included in that sentence, 
e.g [[(2,9), [1,2]], [(30,45), [1,1,2]], [(50,67), [3,3]],...]
"""

def shared_sentences(indices):
  characters = []
  zeros = np.where(indices == 0)[0]
  for i in range(np.shape(indices[:,1])[0]-1):
    if indices[:,1][i] == 0 and indices[:,1][i+1] != 0 and indices[:,1][i+2] != 0:
      characters.append([(indices.item(i,0)+1 , indices.item(zeros[np.where(zeros == i)[0][0] + 1],0)+1) , indices[i+1:zeros[np.where(zeros == i)[0][0] + 1],1].tolist()])
  return characters


# In[6]:


"""
Input: List of 'shared' sentence beginning and end indices, with corresponding
numbers representing the characters involved in the sentence
Output: Removes all entries where the same character appears more than once
with no other character in the sentence, e.g. [[(2,9), [1,2]], [(30,45), [1,1,2]],...]
"""

def remove_same(sentences):
  remove = []
  count = 0
  for i in sentences:
    if len(i[1]) == 2 and i[1][0][0] == i[1][1][0]:
      remove.append(count)
    count += 1
  remove.reverse()
  for i in remove:
    sentences.pop(i)
  return sentences


# In[7]:


"""
convert list text to normal document text
"""

def list_to_doc(text):
  string = ' '.join(word for word in text)
  string = string.replace(' .', '.')
  string = string.replace(' ,', ',')
  string = string.replace(' !', '!')
  string = string.replace(' ?', '?')
  string = string.replace(' ;', ';')
  string = string.replace(' :', ':')
  string = string.replace('[ ', '[')
  string = string.replace(' ]', ']')
  string = string.replace(" ' ", "'")
  string = string.replace("s'", "s' ")
  return string

# In[9]:


"""
Input: dictionary of character names with corresponding number representing character, list of clusters, original list (not tokenized)
Output: List of clusters, streamlined so there are only mention indices and 1 number representing character
can account for when multiple clusters refer to the same character
uses dictionaries instead of lists, more efficient
Characters: Dictionary (key = name, value = number)
Output: Dictionary [[(indices), (indices),..., id-number], ..., ...]
"""
def augment_clusters(clusters, characters):
  augmented_clusters = {}
  for cluster in clusters:
    check_in = False
    check_first = False
    for i in cluster:
      if i[1] in characters:
        check_in = True
        number = characters[i[1]]
        #character = i[1]
        if i[1] not in augmented_clusters:
          check_first = True
        break
    if check_in:
      if check_first:
        augmented_clusters[number] = [i[0] for i in cluster]
      else:
        augmented_clusters[number].extend([i[0] for i in cluster])

  list_augmented_clusters = list(augmented_clusters.items())

  def tuple_to_list(tupl):
    new_list = [i for i in tupl]
    return new_list

  def sort_tuples(key_value):
    key_value[1].sort(key=lambda i:i[0],reverse=False)
    return key_value

  def start_to_end(some_list):
    new_list = some_list[1]
    new_list.append(some_list[0])
    return new_list

  augmented_list = [start_to_end(sort_tuples(tuple_to_list(i))) for i in list_augmented_clusters]
  
  return augmented_list


# In[10]:


"""
Input: list of characters (their numbers and name), [[id, character], ...]
output: character pair encoding dictionary and corresponding character pairs for reference
and a dictionary of character pair numbers with empty list as their value
dictionary_1 {5: (Achilles, Agamemnon), ...}
dictionary_2 {5 : [], ...}
"""
def character_pair_encoder(characters):
  n = len(characters)
  numbers = list(range(n+1))
  numbers.remove(0)

  dictionary_1 = {}
  dictionary_2 = {}

  for i in numbers:
    for j in numbers[i:]:
      dictionary_1[i**2 + j **2] = (characters[i-1][1], characters[j-1][1])
      dictionary_2[i**2 + j **2] = []

  return dictionary_1, dictionary_2


# In[11]:


"""
Goes through the shared sentences with their shared character ids
Assigns the sentence indices to a dictionary of all character pairs based on the encoding above
Input: shared - , pair_dict - 
Output: 
"""

def assign_to_dict(shared, pair_dict):
  for i in shared:
    if len(i[1]) == 2:
      pair_dict[i[1][0][0]**2 + i[1][1][0]**2].extend([i])
    else:
      for a in list(itertools.combinations(set(np.asarray(i[1])[:,0]), 2)):
        pair_dict[a[0]**2 + a[1]**2].extend([i])
  return pair_dict


# In[ ]:


"""
function to give relative list index of character in sentence
Input: sentence indices, character indices in entire document
Output: character indices relative to sentence
"""

def rel_indices(sentence_inds, char_inds):
  return (char_inds[0] - sentence_inds[0], char_inds[1]- sentence_inds[0])


"""
CHANGE
Converts a pair of list indices to a pair of corresponding string indices
"""
def list_to_str_index(doc, indices):
  return (len(list_to_doc(doc[:indices[0]+1]))-len(doc[indices[0]]), len(list_to_doc(doc[:indices[1]+1])))

"""
CHANGE
Function that given the sentence list indices, 
and the character mention indices relative to the sentence
Returns the sentence as a string, and the character mention
indices as string indices
"""
def convert_dict(dic, document):
  for i in dic:
    for n in  range(len(dic[i])):
      print(dic[i][n][0][0])
      print(dic[i][n][0][1])
      list_text = document[dic[i][n][0][0]: dic[i][n][0][1]]
      text = list_to_doc(list_text)
      
      dic[i][n] = [text, [[a[0], list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1])), 
                           text[list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1]))[0]:list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1]))[1]]] for a in dic[i][n][1]]]
  return dic


"""
converts the dictionary of list_document indices above to
dictionary of shared sentences in string format
"""

def dict_ind_to_sentence(dictionary, document):
  for i in dictionary:
    dictionary[i] = [list_to_doc(document[a[0]:a[1]]) for a in dictionary[i]]
  return dictionary


def total_function(clusters, characters_dict, characters_list, list_text):
  augmented_clusters = augment_clusters(clusters, characters_dict)
  punctuation_indices = punctuation_indices(list_text)
  augmented_indices = add_entity_indices(punctuation_indices, augmented_clusters)
  shared = shared_sentences(augmented_indices)
  cleaned_shared = remove_same(shared)

  encoding_dict, shared_sentence_dict = character_pair_encoder(characters_list)
  pair_dict = assign_to_dict(cleaned_shared, shared_sentence_dict)
  pair_dict = convert_dict(pair_dict, list_text)

  return encoding_dict, pair_dict

