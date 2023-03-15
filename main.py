#THIS IS WHAT YOU RUN TO GET YOUR POEMS:
from models import HiddenMarkovModel as HMM
import models
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from collections import Counter
# from embeding_utils import make_embed_model  , convert_text
# from word2vec import word2vec

def load_and_tokenize_shakespeare():
    '''This Loads data from the shakespeare.txt file.
    It should return a list where each element is a poem.
    That poem should be tokenized (ie it is a list where
    each element is a token. Each poem starts with
    the <START> token and ends with the <STOP> token.)'''
    print("Reading Data Shakespeare")
    f = open('data/shakespeare.txt','r')
    text = f.read()
    text = text.lower()
    poems = text.split('\n\n')
    token_list = []
    for i , poem in enumerate(poems):
        print(f"\r Poem {i}", end='')
        poem = poem.replace(',', ' , ')
        poem = poem.replace(':', ' : ')
        poem = poem.replace('?', ' ? ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace('!', ' ! ')
        poem = poem.replace('\n', ' \n ')
        poem = poem.replace('.',' . ')
        poem = poem.replace('(','')
        poem = poem.replace(')','')
        tokens = poem.split(' ')
        tokens.insert(0,'<START>')
        tokens.append('<STOP>')
        tokens = [x for x in tokens if (x != '' and not x.isdigit())]
        token_list.append(tokens)

    print("\n Finished loading")
    print("\n Making Vocabulary")
    mega_list = []
    for tokens in token_list:
        mega_list.extend(tokens)
    vocab_counts = Counter(mega_list)
    vocab_list = sorted(vocab_counts.items(), key=lambda x: x[1])
    vocab_list = [x for (x,y) in vocab_list]
    vocab_list.reverse()
    #Returns a list of tokens
    # and also a vocabulary sorted by the frequency of the tokens
    return token_list , vocab_list

def load_and_tokenize_spencer():
    '''This Loads data from the shakespeare.txt file.
    It should return a list where each element is a poem.
    That poem should be tokenized (ie it is a list where
    each element is a token. Each poem starts with
    the <START> token and ends with the <STOP> token.)'''
    print("Reading Data Spencer")
    f = open('data/spenser.txt','r')
    text = f.read()
    text = text.lower()
    poems = text.split('\n\n')
    token_list = []
    for i , poem in enumerate(poems):
        print(f"\r Poem {i}", end='')
        poem = poem.replace(',', ' , ')
        poem = poem.replace(':', ' : ')
        poem = poem.replace('?', ' ? ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace('!', ' ! ')
        poem = poem.replace('\n', ' \n ')
        poem = poem.replace('.',' . ')
        poem = poem.replace('(','')
        poem = poem.replace(')','')
        tokens = poem.split(' ')
        tokens.insert(0,'<START>')
        tokens.append('<STOP>')
        tokens = [x for x in tokens if (x != '' and x != '\t')]
        token_list.append(tokens)

    print("\n Finished loading")
    print("\n Making Vocabulary")
    mega_list = []
    for tokens in token_list:
        mega_list.extend(tokens)
    vocab_counts = Counter(mega_list)
    vocab_list = sorted(vocab_counts.items(), key=lambda x: x[1])
    vocab_list = [x for (x,y) in vocab_list]
    vocab_list.reverse()
    #Returns a list of tokens
    # and also a vocabulary sorted by the frequency of the tokens
    return token_list , vocab_list


def load_and_tokenize_collated():
    '''This Loads data from the shakespeare.txt file.
    It should return a list where each element is a poem.
    That poem should be tokenized (ie it is a list where
    each element is a token. Each poem starts with
    the <START> token and ends with the <STOP> token.)'''
    print("Reading Data Spencer")
    f = open('data/collated_sonnets.txt','r')
    text = f.read()
    text = text.lower()
    poems = text.split('\n\n')
    token_list = []
    for i , poem in enumerate(poems):
        print(f"\r Poem {i}", end='')
        poem = poem.replace(',', ' , ')
        poem = poem.replace(':', ' : ')
        poem = poem.replace('?', ' ? ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace(';', ' ; ')
        poem = poem.replace('!', ' ! ')
        poem = poem.replace('\n', ' \n ')
        poem = poem.replace('--', ' , ')
        poem = poem.replace('.',' . ')
        poem = poem.replace('(','')
        poem = poem.replace(')','')
        tokens = poem.split(' ')
        tokens.insert(0,'<START>')
        tokens.append('<STOP>')
        tokens = [x for x in tokens if (x != '' and x != '\t')]
        token_list.append(tokens)

    print("\n Finished loading")
    print("\n Making Vocabulary")
    mega_list = []
    for tokens in token_list:
        mega_list.extend(tokens)
    vocab_counts = Counter(mega_list)
    vocab_list = sorted(vocab_counts.items(), key=lambda x: x[1])
    vocab_list = [x for (x,y) in vocab_list]
    vocab_list.reverse()
    #Returns a list of tokens
    # and also a vocabulary sorted by the frequency of the tokens
    return token_list , vocab_list



def obs_map(tokens, vocab_list):
    '''HMM can only handle integer sequences, so we
    map every word to an integer'''
    mapped_tokens = []
    for token in tokens:
        mapped_tokens.append(vocab_list.index(token))
    return mapped_tokens

def reverse_obs_map(tokens,vocab_list):
    '''For unmapping words back to real tokens'''
    unmapped_tokens = []
    for token in tokens:
        unmapped_tokens.append(vocab_list[token])
    return unmapped_tokens

class ObsMapEmbed:
    #This class is for use in the LSTM.
    def __init__(self,vocab):
        self.vocab = vocab

    def embed(self,tokens):
        #Takes a list of words and embeds them.
        return th.nn.functional.one_hot(th.tensor(obs_map(tokens,self.vocab)),num_classes = len(self.vocab))

    def unembed(self,tokens):
        # recieves a list of one hot embedded tokens and then turns them into words.

        return reverse_obs_map(tokens,self.vocab)

def sample_sentence(hmm, vocab_list, max_words=1000):
    # Get reverse map.

    # Sample and convert sentence.
    end_token = obs_map(['<STOP>'],vocab_list)[0]
    start_token = obs_map(['<START>'],vocab_list)[0]
    emission, states = hmm.generate_emission(max_words,start_token,end_token=end_token)
    sentence = emission
    sentence = reverse_obs_map(sentence,vocab_list)
    output = ' '.join(sentence).capitalize()
    output = output.replace(' <start>','')
    output = output.replace(' .','.')
    output = output.replace(' ?','?')
    output = output.replace(' ,',',')
    output = output.replace(' :',':')
    output = output.replace(' ;',';')
    output = output.replace(' !','!')
    output = output.replace(' \n ','\n')
    print(output)
    #TODO figure out how to output the sates as well,
    #So that they also end at <STOP>
    return output


def train_HMM(tokens, model):
    '''train an HMM on a set of tokens
    Where the tokens are of the form outputed by load_and_tokenize'''
    #TODO: RN the HMM can only handle numbers
    #TODO JUST USE THEIR CODE.
    pass


def LSTM_loss(y,y_hat):
    return th.mean(-th.log(th.cosine_similarity(y,y_hat)))

def train_LSTM(model,X,batch_size,num_epochs,lr):
    '''

    :param model: an LSTM model
    :param X: a list of sequences
    :param batch_size:
    :param num_epochs:
    :return:
    '''
    Y = [model.embedding_model.embed(x[1:]) for x in X]
    X = [model.embedding_model.embed(x[:-1]) for x in X]
    train_dataset = data_utils.TensorDataset(th.nn.utils.rnn.pad_sequence(X).permute([1,0,2]).float(), th.nn.utils.rnn.pad_sequence(Y).permute([1,0,2]).float())
    data = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fun = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(num_epochs):
        avg_loss = 0
        counter = 0
        for x, y in data:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fun(th.flatten(y_hat,0,1), th.flatten(y,0,1))
            loss.backward()
            optimizer.step()
            avg_loss += loss
            counter += 1
        print(f"epoch - {epoch} avg_loss: {avg_loss / counter}")
        model.sample_sentence(max_tokens=15)
    print("Finished")
    return model


def visualize_HMM(model):
    '''Do wordclouds and stuff'''
    #TODO
    pass

def experiment():
    '''Makes the Model, Trains the Model, Outputs some poem
    examples, then outputs some visualizations.'''
    #Load the tokens from Shakespeare
    token_list , vocab_list = load_and_tokenize_spencer()
    #Then map them s.t. each word is an integer
    mapped_token_list = []
    for poem in token_list :
        mapped_token_list.append(obs_map(poem,vocab_list))
    model = models.unsupervised_HMM(mapped_token_list,16,30)
    for i in range(40):
        print(f"\n\nSentence {i}\n")
        sample_sentence(model,vocab_list,300)
    print("END OF PROGRAM")

def lstm_experiment():
    token_list, vocab_list = load_and_tokenize_collated()
    embed = ObsMapEmbed(vocab_list)
    model = models.LSTM_Poet(400,embed,len(vocab_list))
    model = train_LSTM(model, token_list, 16, 30,.01)
    print("\n\n\n\n final sample poems:")
    for i in range(40):
        print(f"\n\nPoem {i}\n" )
        model.sample_sentence(500)

if __name__ == '__main__':
    lstm_experiment()
