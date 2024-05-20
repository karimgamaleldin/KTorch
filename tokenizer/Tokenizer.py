import json
from typing import List

class Tokenizer:
  def __init__(self, pad_token: str = '<PAD>', unk_token: str = '<UNK>', bos_token: str = '<BOS>', eos_token: str = '<EOS>',  mask_token: str = '<MASK>' , end_of_word_token: str = '</w>', ignore_case: bool = False):
    '''
    Create a new Tokenizer object
    
    Parameters:
    pad_token (str): Padding token
    unk_token (str): Unknown token
    bos_token (str): Beginning of sentence token
    eos_token (str): End of sentence token
    mask_token (str): Mask token
    end_of_word_token (str): End of word token
    '''
    self.pad_token = pad_token  
    self.unk_token = unk_token
    self.bos_token = bos_token
    self.eos_token = eos_token
    self.mask_token = mask_token
    self.end_of_word_token = end_of_word_token
    self.ignore_case = ignore_case
    self.vocab = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3, self.mask_token: 4, self.end_of_word_token: 5}
    self.idx2token = {v: k for k, v in self.vocab.items()}
    
  def encode(self, text: str) -> List[int]:
    '''
    Convert text to a list of ids
    '''
    return [self.vocab.get(token, self.vocab[self.unk_token]) for token in text.split()]
  
  def decode(self, tokens) -> str:
    '''
    Convert a list of ids to text
    '''
    return "".join([self.idx2token[token] for token in tokens])
  
  def tokenize(self, text: str) -> List[str]:
    '''
    Convert text to a list of tokens
    '''
    return text.split()
  
  def detokenize(self, tokens: List[str]) -> str:
    '''
    Convert a list of tokens to text
    '''
    return " ".join(tokens)
  
  def save(self, path: str):
    '''
    Save the vocabulary to a file
    '''
    with open(path, "w") as f:
      json.dump(self.vocab, f)

  def train(self, corpus: List[str], vocab_size: int):
    '''
    Train the tokenizer on a corpus, implemented in the subclass
    '''
    raise NotImplementedError("Tokenizer must implement a train method")
  
  def insert_to_vocab(self, token: str):
    '''
    Insert a token to the vocabulary
    '''
    self.vocab[token] = len(self.vocab)
    self.idx2token[self.vocab[token]] = token
  
  def insert_end_of_word(self, sentence: str) -> str:
    '''
    Insert the end of word token at the end of each word
    '''
    removed_spaces = sentence.split()
    sentence = [word + self.end_of_word for word in removed_spaces]
    return " ".join(sentence)
  
  def add_characters_to_vocab(self, corpus: List[str]):
    '''
    Add characters to the vocabulary
    '''
    for sentence in corpus:
      for token in sentence.split():
        for char in token:
          if self.ignore_case:
            char = char.lower()
          if char not in self.vocab:
            self.insert_to_vocab(char)