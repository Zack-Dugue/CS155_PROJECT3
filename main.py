#THIS IS WHAT YOU RUN TO GET YOUR POEMS:
from models import HiddenMarkovModel as HMM

def load_and_tokenize_shakespeare():
    '''This Loads data from the shakespeare.txt file.
    It should return a list where each element is a poem.
    That poem should be tokenized (ie it is a list where
    each element is a token. Each poem starts with
    the <START> token and ends with the <STOP> token.'''
    pass
    #TODO ZACK WILL DO THIS
def load_and_tokenize_spenser():
    '''This Loads data from the spenser.txt file.
    It should return a list where each element is a poem.
    That poem should be tokenized (ie it is a list where
    each element is a token. Each poem starts with
    the <START> token and ends with the <STOP> token.'''
    pass
    #TODO ALICIA WILL DO THIS (BASED ON ZACK'S IMPLEMENTATION PROBABLY)

def train_HMM(tokens, model):
    '''train an HMM on a set of tokens
    Where the tokens are of the form outputed by load_and_tokenize'''
    #TODO JUST USE THEIR CODE.
    pass

def train_LSTM(tokens, model):
    '''train an LSTM on a set of tokens
    Where the tokens are of the form outputed by load_and_tokenize'''
    #TODO: ZACK WILL DO THIS
    pass

def visualize_HMM(model):
    '''Do wordclouds and stuff'''
    #TODO
    pass

def experiment():
    '''Makes the Model, Trains the Model, Outputs some poem
    examples, then outputs some visualizations.'''
    #TODO
    pass

if __name__ == '__main__':
    experiment()
