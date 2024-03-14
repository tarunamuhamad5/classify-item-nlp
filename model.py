import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


class model_predict:
    def __init__(self):
        self.model_for_sales = pickle.load(open('model/model-for-sales.sav','rb'))
        self.vect_for_sales = pickle.load(open('model/vect-for-sales.sav','rb'))
        
        self.model_community = pickle.load(open('model/model-community.sav','rb'))
        self.vect_community = pickle.load(open('model/vect-community.sav','rb'))
        
        self.model_housing = pickle.load(open('model/model-housing.sav','rb'))
        self.vect_housing = pickle.load(open('model/vect-housing.sav','rb'))
        
        self.model_services = pickle.load(open('model/model-services.sav','rb'))
        self.vect_services = pickle.load(open('model/vect-services.sav','rb'))
        
    def predict_for_sales(self,text):
        text_dtm = self.vect_for_sales.transform(text)
        return self.model_for_sales.predict(text_dtm)
    
    def predict_community(self,text):
        text_dtm = self.vect_community.transform(text)
        return self.model_community.predict(text_dtm)
    
    def predict_housing(self,text):
        text_dtm = self.vect_housing.transform(text)
        return self.model_housing.predict(text_dtm)
    
    def predict_services(self,text):
        text_dtm = self.vect_services.transform(text)
        return self.model_services.predict(text_dtm)
    
    def predict(self,text,section):
        text =re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
        if section =='for-sale':
            return self.predict_for_sales([text])
        elif section =='housing':
            return self.predict_housing([text])
        elif section =='community':
            return self.predict_community([text])
        elif section =='services':
            return self.predict_services([text])
        
        
if __name__ == '__main__':
    test = model_predict()
    print(test.predict('silver platinum gameboy pocket immaculate','for-sale'))