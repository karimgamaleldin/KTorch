from .Tokenizer import Tokenizer
from typing import List

class BPE(Tokenizer):
  ''' 
  Byte Pair Encoding (BPE) tokenization algorithm
  '''
  def __init__(self, pad_token: str = '<PAD>', unk_token: str = '<UNK>', bos_token: str = '<BOS>', eos_token: str = '<EOS>',  mask_token: str = '<MASK>' , end_of_word_token: str = '</w>', ignore_case: bool = False):
    super().__init__(pad_token, unk_token, bos_token, eos_token, mask_token, end_of_word_token, ignore_case)


  def train(self, corpus, vocab_size):
    ''' 
    Train the BPE model on a corpus

    Parameters:
    corpus (List[str]): List of strings to train the model on
    vocab_size (int): Size of the vocabulary
    '''
    # Add bytes to the vocabulary
    self.add_first_256_bytes_to_vocab()

    # Preprocess the corpus
    self.preprocess_corpus(corpus)

    # Create a vocabulary with the characters in the corpus
    for i in range(vocab_size):
      max_pair = self.get_max_pairs(corpus) # Get the most frequent pair of characters
      self.insert_to_vocab(max_pair) # Add the token to the vocabulary
      token_id = self.vocab[max_pair] # Get the token id
      for i in range(len(corpus)):
        corpus[i] = self.merge_pair(corpus[i], max_pair, token_id) # Merge the most frequent pair of characters in each sentence

    print("BPE tokenizer trained successfully!")

  def convert_to_utf8(self, text: str) -> List[int]:
    '''
    Convert text to UTF-8 encoding and return a list of integers
    '''
    tokens = text.encode('utf-8')
    return list(map(int, tokens))
  
  def get_max_pairs(self, corpus) -> tuple:
    '''
    Count the number of times each pair of characters appears in the corpus
    '''
    counts = {}
    for sentence in corpus:
      for word in sentence:
        for pair in zip(word, word[1:]):
          counts[pair] = counts.get(pair, 0) + 1
    max_pair = max(counts, key=counts.get)
    return max_pair
  
  def merge_pair(self, sentence: List[List[int]], pair: tuple, token_id: int) -> str:
    '''
    Merge the most frequent pair of characters in the sentence

    Parameters:
    sentence (List[int]): List of integers representing the sentence
    pair (tuple): Pair of characters to merge
    '''
    new_sentence = []
    for word in sentence:
      new_word = []
      i = 0
      while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1]) == pair: # Merge the pair
          new_word.append(token_id)
          i += 2
        else: # Keep the character
          new_word.append(word[i]) 
          i += 1
      new_sentence.append(new_word)
        
    return new_sentence
  
  def add_first_256_bytes_to_vocab(self):
    '''
    Add the first 256 bytes to the vocabulary
    '''
    for i in range(256):
      self.insert_to_vocab(i)


  def preprocess_corpus(self, corpus: List[str]) -> List[str]:
    '''
    Preprocess the corpus by adding the end of word token to each word
    '''
    for i in range(len(corpus)):
      sentence = corpus[i].split()
      new_sentence = [] 
      for word in sentence:
        encoded = self.convert_to_utf8(word) # Convert the word to UTF-8 encoding
        encoded = [self.vocab.get(token, self.vocab[self.unk_token]) for token in encoded] # Convert the bytes to a list of ids
        encoded += [self.vocab[self.end_of_word_token]]
        new_sentence.append(encoded)
      corpus[i] = new_sentence

