'''This exists in order to get some useful word embeddings from the data.'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np

##########################
# Helper functions/classes
##########################

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

def get_similarity(v1, v2):
    """ Returns the cosine of the angle between vectors v1 and v2. """
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.dot(v1_unit, v2_unit)


def load_word_list(path):
    """
    Loads a list of the words from the file at path <path>, removing all
    non-alpha-numeric characters from the file.
    """
    with open(path) as handle:
        # Load a list of whitespace-delimited words from the specified file
        raw_text = handle.read().strip().split()
        # Strip non-alphanumeric characters from each word
        alphanumeric_words = map(lambda word: ''.join(char for char in word if char.isalnum()), raw_text)
        # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
        alphanumeric_words = filter(lambda word: len(word) > 0, alphanumeric_words)
        # Convert each word to lowercase and return the result
        return list(map(lambda word: word.lower(), alphanumeric_words))

def generate_onehot_dict(word_list):
    """
    Takes a list of the words in a text file, returning a dictionary mapping
    words to their index in a one-hot-encoded representation of the words.
    """
    word_to_index = {}
    i = 0
    for word in word_list:
        if word not in word_to_index:
            word_to_index[word] = i
            i += 1
    return word_to_index

def most_similar_pairs(weight_matrix, word_to_index):
    """
    For each word a in our vocabulary, computes the most similar word b to a, along with the
    cosine similarity of a and b.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

    Returns:
        A list of WordPair instances sorted in decreasing order of similarity,
        one representing each word <vocab_word> and its most similar word.
    """
    word_to_feature_repr = get_word_to_feature_repr(weight_matrix, word_to_index)
    result = []
    for word in word_to_feature_repr:
        result.append(most_similar_word(word_to_feature_repr, word))
    return sort_by_similarity(result)

def most_similar_word(word_to_feature_repr, input_word):
    """
    Given a dictionary mapping words to their feature representations (word_to_feature_repr),
    returns the a WordPair instance corresponding to the word
    whose feature vector is most similar to the feature representation of the
    passed-in word (input_word).
    """
    best_word = None
    best_similarity = 0
    input_vec = word_to_feature_repr[input_word]
    for word, feature_vec in word_to_feature_repr.items():
        similarity = get_similarity(input_vec, feature_vec)
        if similarity > best_similarity and np.linalg.norm(feature_vec - input_vec) != 0:
            best_similarity = similarity
            best_word = word
    return WordPair(input_word, best_word, best_similarity)

def get_word_to_feature_repr(weight_matrix, word_to_index):
    """
    Returns a dictionary mapping each word in our vocabulary to its one-hot-encoded
    feature representation.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.
    """
    assert(weight_matrix is not None)
    word_to_feature_repr = {}
    for word, one_hot_idx in word_to_index.items():
        word_to_feature_repr[word] = weight_matrix[one_hot_idx]
    return word_to_feature_repr


def get_word_repr(word_to_index, word):
    """
    Returns one-hot-encoded feature representation of the specified word given
    a dictionary mapping words to their one-hot-encoded index.

    Arguments:
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        word:          Word whose feature representation we wish to compute.

    Returns:
        feature_representation:     Feature representation of the passed-in word.
    """
    unique_words = word_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_words))
    feature_representation[word_to_index[word]] = 1
    return feature_representation


def generate_traindata(word_list, word_to_index = None, window_size=4):
    """
    Generates training data for Skipgram model.

    Arguments:
        word_list:     Sequential list of words (strings).
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        window_size:   Size of Skipgram window. Defaults to 2
                       (use the default value when running your code).

    Returns:
        (trainX, trainY):     A pair of matrices (trainX, trainY) containing training
                              points (one-hot-encoded vectors) and their corresponding output_word
                              (also one-hot-encoded vectors)

    """
    trainX = []
    trainY = []
    for i , word in enumerate(word_list):
        word_repr = word
        for j in range(-window_size, window_size):
            if j == 0:
                continue
            try:
                trainY.append(word_list[i + j])
                trainX.append(word_repr)
            except Exception:
                continue
        assert(len(trainX) == len(trainY))

    return torch.tensor(np.array(trainX)), torch.tensor(np.array(trainY))

class EmbedModel(nn.Module):
    def __init__(self,vocab_size ,embed_dim = 10):
        super(EmbedModel, self).__init__()
        self.wI = nn.Linear(vocab_size,embed_dim)
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.wO = nn.Linear(embed_dim,vocab_size)
        self.out = nn.Softmax()
    def forward(self, x,inference=False):
        embedding = self.wI(x)
        embedding = self.norm(embedding)
        prediction = self.wO(embedding)
        if inference:
            prediction = self.softmax(prediction)
        return prediction, embedding
    def embed(self,x,sigma=0):
        embedding = self.wI(x)
        embedding = self.norm(embedding)
        if sigma != 0:
            embedding += torch.randn_like(embedding)*sigma
        return embedding
    def de_embed(self,x):
        dembed = self.wI(x)
        return self.norm(x)


class EmbedModel2(nn.Module):
    def __init__(self,vocab_size ,embed_dim = 10):
        super(EmbedModel2, self).__init__()
        self.wI0 = nn.Linear(vocab_size,4*embed_dim)
        self.act1 = nn.GELU()
        self.wI = nn.Linear(4*embed_dim,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.wO0 = nn.Linear(embed_dim,4*embed_dim)
        self.act2 = nn.GELU()
        self.wO = nn.Linear(4*embed_dim,vocab_size)
        self.out = nn.Softmax()
    def forward(self, x,inference=False):
        x = self.wI0(x)
        x = self.act1(x)
        embedding = self.wI(x)
        embedding = self.norm(embedding)
        embedding = self.wO0(embedding)
        prediction = self.wO(embedding)
        if inference:
            prediction = self.softmax(prediction)
        return prediction, embedding

def train(trainX, trainY,model, lr = .01, batch_size = 100,num_epochs=100):
    """Returns a trained model"""
    train_dataset = data_utils.TensorDataset(trainX.float(), trainY)
    data = data_utils.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
    optimizer = optim.Adam(model.parameters(),lr)
    loss_fun = nn.CrossEntropyLoss(reduction="mean")
    for epoch in range(num_epochs):
        avg_loss = 0
        counter = 0
        for x,y in data:
            optimizer.zero_grad()
            y_hat, _ = model(x)
            loss = loss_fun(y_hat,y)
            loss.backward()
            optimizer.step()
            avg_loss += loss
            counter += 1
        print(f"epoch - {epoch} avg_loss: {avg_loss/counter}")
    print("Finished")
    return model




def find_most_similar_pairs(filename, num_latent_factors):
        """
        Find the most similar pairs from the word embeddings computed from
        a body of text

        Arguments:
            filename:           Text file to read and train embeddings from
            num_latent_factors: The number of latent factors / the size of the embedding
        """
        # Load in a list of words from the specified file; remove non-alphanumeric characters
        # and make all chars lowercase.
        sample_text = load_word_list(filename)

        # Create word dictionary
        word_to_index = generate_onehot_dict(sample_text)
        print("Textfile contains %s unique words" % len(word_to_index))
        # Create training data
        trainX, trainY = generate_traindata(sample_text, word_to_index)

        # vocab_size = number of unique words in our text file. Will be useful
        # when adding layers to your neural network
        vocab_size = len(word_to_index)

        model = EmbedModel(vocab_size,embed_dim = num_latent_factors)
        model = train(trainX,trainY,model)
        # set weights variable below
        weights = model.wI.weight
        print(f"input weight dimensions = {weights.size()}")
        weights_2 = model.wO.weight
        print(f"output weight dimensions = {weights_2.size()}")
        # Find and print most similar pairs
        similar_pairs = most_similar_pairs(torch.asarray(weights).T, word_to_index)
        for pair in similar_pairs[:30]:
            print(pair)

def make_embedding(token_data, vocabulary,num_latent_factors=256):
    #Assumes that the token Data has already been converted into a one hot form.
    model = EmbedModel(len(vocabulary), embed_dim=num_latent_factors)
    trainX, trainY = generate_traindata(token_data, vocabulary)
    embed_model = train(trainX, trainY, model)
    return embed_model

