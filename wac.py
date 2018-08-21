
import numpy as np
from sklearn import linear_model
import pickle
from operator import itemgetter

class WAC:
    
    def __init__(self,name,classifier_spec=(linear_model.LogisticRegression,{'penalty':'l2'}), compose_method='prod'):
        '''
        name: name of model for persistance and loading
        compose_method: prod, avg, sum
        classifier_spec: some kind of scikit classifier + arguments as a tuple
        '''
        self.wac = {}
        self.model_name=name
        self.classifier_spec = classifier_spec
        self.current_utt = {}
        self.compose_method = compose_method
        
    def vocab(self):
        return self.wac.keys()
    
    def add_observation(self, word, features, label):
        if word not in self.wac: self.wac[word] = list()
        self.wac[word].append((features, label))
    
    def add_multiple_observations(self, word, features, labels):
        for f,p in zip(features,labels):
            self.add_observation(word, f, p)
    
    def load_model(self):
        with open('{}.pickle'.format(self.model_name), 'rb') as handle:
            self.wac = pickle.load(handle)
    
    def persist_model(self):
        with open('{}.pickle'.format(self.model_name), 'wb') as handle:
            pickle.dump(self.wac,handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def train(self, min_obs=4):
        classifier, classf_params = self.classifier_spec
        nwac = {}
        for word in self.wac:
            if len(self.wac[word]) < min_obs: continue
            this_classf = classifier(**classf_params)
            X,y = zip(*self.wac[word])
            this_classf.fit(X,y)
            nwac[word] = this_classf
        self.wac = nwac
        
    def get_current_prediction_state(self):
        return self.current_utt
    
    def get_predicted_intent(self):
        return max(self.get_current_prediction_state(), key=itemgetter(1))
    
    def compose(self, predictions):
        if self.current_utt == {}:
            self.current_utt = predictions 
            return self.current_utt
        
        composed = []
        for one,two in zip(predictions,self.current_utt):
            i1,p1=one
            i2,p2=two
            if i1 != i2:
                print('intent mismatch! {} != {}'.format(i1,i2))
                continue
            
            res = 0
            if self.compose_method == 'sum':
                res = p1 + p2
            if self.compose_method == 'prod':
                res = p1 * p2
            if self.compose_method == 'avg':
                res = p1 * p2 / 2.0
            
            composed.append((i1,res))
            
        self.current_utt = composed
        return self.current_utt
        
    
    def add_increment(self, word, context):
        predictions  = self.proba(word, context)
        return self.compose(predictions)
    
    def proba(self, word, context):
        intents,feats = context
        if word not in self.wac: return None # todo: return a distribution of all zeros?
        predictions = list(zip(intents,self.wac[word].predict_proba(np.array(feats))[:,1]))
        return predictions
    
    def new_utt(self):
        self.current_utt = {} 
    
