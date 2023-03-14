'''This exists in order to get some useful word embeddings from the data.'''

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import re
import numpy as np
from word2vec import word2vec
##########################
# Helper functions/classes
##########################

def convert_text():
    f = open("data/shakespeare.txt", 'r')
    text = f.read()
    f.close()
    text = re.sub('[0123456789]', '', text)
    text = text.replace("\n\n", "<START>\n\n<STOP>")
    text = text.replace("." , " . ")
    text = text.replace("," , " , ")
    text = text.replace(":" , " : ")
    text = text.replace("!" , " ! ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(";", " ; ")
    text = text.replace("?" , " ? ")
    f = open("data/tokenized_shakespeare.txt", 'w')
    text = '<START> ' + text + ' <STOP>'
    f.write(text)
    f.close()

class WordPair:
    """
    Class representing a pair of words in our vocabulary, along with the cosine similarity
    of the two words.
    """
    def __init__(self, firstWord, secondWord, similarity):
        """
        Initializes the WordPair given two words (strings) and their similarity (float).
        """
        # Ensure that our pair consists of two distinct words
        assert(firstWord != secondWord)
        self.firstWord = firstWord
        self.secondWord = secondWord
        self.similarity = similarity

    def __repr__(self):
        """
        Define the string representation of a WordPair so that a WordPair instance x
        can be displayed using print(x).
        """
        return "Pair(%s, %s), Similarity: %s"%(self.firstWord, self.secondWord, self.similarity)


def sort_by_similarity(word_pairs):
    """
    Given a list of word pair instances, returns a list of the instances sorted
    in decreasing order of similarity.
    """
    return sorted(word_pairs, key=lambda pair: pair.similarity, reverse=True)



class EmbedModel3(th.nn.Module):
    def __init__(self, embeddings : dict):
        super(EmbedModel3, self).__init__()
        self.embed_dict = embeddings
        self.values = th.stack(list(embeddings.values()))

    def embed(self, seq):
        # Seq is a list of tokens
        embedded_seq = []
        for token in seq:
            embedded_seq.append(self.embed_dict[token])
        return th.stack(embedded_seq)

    def unembed(self, seq):
        word = th.multinomial(th.softmax(th.matmul(seq, self.values)), num_samples=1)
        return self.embed_dict.keys()[word]

    def get_similarities(self):
        word_similarity_list = []
        q = len(self.embed_dict)*len(self.embed_dict) - len(self.embed_dict)
        counter = 0
        for wordi , embedi in self.embed_dict.items():
            for wordj , embedj in self.embed_dict.items():
                if wordi == wordj:
                    continue
                counter += 1
                print(f"\rfraction complete : {counter/q}", end="")
                word_similarity_list.append(WordPair(wordi,wordj,float(th.cosine_similarity(embedi.view([1,-1]),embedj.view([1,-1])))))
        word_similarity_list =  sort_by_similarity(word_similarity_list)
        for wordpair in word_similarity_list[:20]:
            print(wordpair)


def make_embed_model(path="data/tokenized_shakespeare.txt"):
    word2vec(path, 'data/embeddings.txt',
             hs=0, sample=1e-5, negative=5, min_count=1, save_vocab="data/vocab.txt")
    f = open("data/embeddings.txt", 'r')
    embeddings = {}
    lines  = f.readlines()
    for embed in lines[1:]:
        embed = embed.replace('\n','')
        embed = embed.split(' ')
        embeddings.update({embed[0]: th.tensor([float(x) for x in embed[1:-1]])})
    model = EmbedModel3(embeddings)
    # model.get_similarities()
    return model


if __name__ == '__main__':
    convert_text()
    make_embed_model()