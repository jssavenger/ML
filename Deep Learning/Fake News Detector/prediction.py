from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.corpus import stopwords  
import pandas as pd
import pickle
import time

class Model:
    def __init__(self):
        with open('tokenizer.h5', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        self.model = load_model('trained_model.h5') 
        self.punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
        self.stop_words = set(stopwords.words('turkish'))
        

    def optimization(self, user_input):
        """Dataset Optimization Function
                Args:
                    dataset: Dataset for DL model.
                    return (str): Optimizated user input.
        """
        self.stop_words.update(self.punctuation)
        user_input = user_input.lower()
        result = [user for user in user_input if user not in self.punctuation]
        result = ("").join(result)
        return result


    def start(self):
        while True:
            user = input(f"\nUser: ")
            st = time.time()
            cl = self.optimization(user)
            sequences = self.tokenizer.texts_to_sequences([cl])
            padded_data = pad_sequences(sequences, maxlen=1000)
            prediction = self.model.predict(padded_data)
            print(f"\nResult: {prediction}\nTime: {(time.time() - st)}\n")
    
    def runner(self):
        self.start()
        
model = Model()
model.runner()