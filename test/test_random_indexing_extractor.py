import os
import sys
import codecs
from sklearn.externals import joblib

import unittest

import json
from digRandomIndexingExtractor.random_indexing_extractor import RandomIndexingExtractor
from digExtractor.extractor_processor import ExtractorProcessor


class DummyModel:

    def __init__(self):
        self.count = 0


    def predict(self, feature_vector):
        self.count = self.count + 1
        return self.count > 1



class TestRandomIndexingExtractor(unittest.TestCase):

    def load_model(self, file_path):
        model_file = os.path.join(os.path.dirname(__file__), file_path)
        model = joblib.load(model_file)

        return model

    def load_features(self, file_path):
        feature_file = os.path.join(os.path.dirname(__file__), file_path)
        feature_model = joblib.load(feature_file)

        return feature_model

    def load_embeddings(self, file_path):
        embeddings = dict()
        embeddings_file = os.path.join(os.path.dirname(__file__), file_path)
        with codecs.open(embeddings_file, 'r', 'utf-8') as f:
            for line in f:
                embedding = json.loads(line)
                for k, v in embedding.items():
                    embeddings[k] = v
    	
    	return embeddings


    def test_random_indexing_extractor_mock(self):
        doc = {"tokenized_text": ['There', 'once', 'was', 'a', 'woman', 'named', 'Mary', 'from', 'the', 'city', 'of', 'Charlotte', 'North', 'Carolina' ],"names": ['Charlotte', 'Mary']}
        e = RandomIndexingExtractor().set_embeddings({}).set_model(DummyModel()).set_metadata({"extractor": "dummy_random_indexing_extractor"})
        ep = ExtractorProcessor().set_input_fields(['tokenized_text', 'names']).set_output_field('filtered_names').set_extractor(e)

        updated_doc = ep.extract(doc)
        # self.assertEquals(updated_doc['filtered_names']['value'], list(['Mary']))


    def test_random_indexing_extractor_actual(self):
    	model = self.load_model("model")
        feature_model = self.load_features("features")
        embeddings = self.load_embeddings("embeddings.jl")
        tokenized_text_2 = ['Los', 'Angeles', 'city', 'of', 'dreams', 'call']
        names_2 = ['dreams', 'Los Angeles', 'call', 'city']
        doc = {"tokenized_text": tokenized_text_2,"names": names_2}
    	e = RandomIndexingExtractor().set_embeddings(embeddings).set_model(model).set_feature_model(feature_model).\
            set_metadata({"extractor": "person_name_random_indexing_extractor"})
    	ep = ExtractorProcessor().set_input_fields(['tokenized_text', 'names']).set_output_field('filtered_names').set_extractor(e)

    	updated_doc = ep.extract(doc)
    	self.assertEquals(updated_doc['filtered_names']['value'], list(['Los Angeles']))

#python -m unittest discover
if __name__ == '__main__':
    unittest.main()