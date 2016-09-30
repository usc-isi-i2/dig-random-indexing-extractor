from digExtractor.extractor import Extractor
import re
import copy 
import types
from itertools import ifilter

class RandomIndexingExtractor(Extractor):

    def __init__(self):
        self.renamed_input_fields = ['tokens', 'values']

    def get_embeddings(self):
        return self.embeddings

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings
        return self

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        return self

    def get_feature_vector(self, tokens, value):
        feature_vector = None
        #TODO
        return feature_vector

    def get_model_prediction(self, feature_vector):
        #TODO
        return self.model.predict(feature_vector)

    def predict(self, tokens, value):
        feature_vector = self.get_feature_vector(tokens, value)
        return self.get_model_prediction(feature_vector)

    def extract(self, doc):
        try:
            values = doc['values']
            tokens = doc['tokens']
            output = list()

            if isinstance(values, basestring):
                values = [values]

            output.extend(ifilter(lambda v: self.predict(tokens,v),iter(values)))
            
            return output
        except:
            return list()

    def get_metadata(self):
        return copy.copy(self.metadata)

    def set_metadata(self, metadata):
        self.metadata = metadata
        return self

    def get_renamed_input_fields(self):
        return self.renamed_input_fields;

