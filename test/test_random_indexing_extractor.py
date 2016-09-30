import os
import sys
import codecs

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
        #TODO
        model = ""

        return model

    def load_embeddings(self, file_path):
        embeddings = list()
        embeddings_file = os.path.join(os.path.dirname(__file__), file_path)
        with codecs.open(embeddings_file, 'r', 'utf-8') as f:
            for line in f:
                embedding = json.loads(line)
                embeddings.extend(embedding)
    	
    	return embeddings


    def test_random_indexing_extractor_mock(self):
        doc = {"tokenized_text": ['There', 'once', 'was', 'a', 'woman', 'named', 'Mary', 'from', 'the', 'city', 'of', 'Charlotte', 'North', 'Carolina' ],"names": ['Charlotte', 'Mary']}
        e = RandomIndexingExtractor().set_embeddings("").set_model(DummyModel()).set_metadata({"extractor": "dummy_random_indexing_extractor"})
        ep = ExtractorProcessor().set_input_fields(['tokenized_text', 'names']).set_output_field('filtered_names').set_extractor(e)

        updated_doc = ep.extract(doc)
        print updated_doc
        self.assertEquals(updated_doc['filtered_names']['value'], list(['Mary']))


    def test_random_indexing_extractor_actual(self):
    	model = self.load_model("model")
        embeddings = self.load_model("embeddings.jl")
        
        doc = {"tokenized_text": ['There', 'once', 'was', 'a', 'woman', 'named', 'Mary', 'from', 'the', 'city', 'of', 'Charlotte', 'North', 'Carolina' ],"names": ['Charlotte', 'Mary']}
    	e = RandomIndexingExtractor().set_embeddings(embeddings).set_model(model).set_metadata({"extractor": "person_name_random_indexing_extractor"})
    	ep = ExtractorProcessor().set_input_fields(['tokenized_text', 'names']).set_output_field('filtered_names').set_extractor(e)

    	updated_doc = ep.extract(doc)
    	#self.assertEquals(updated_doc['filtered_names']['value'], list(['Mary']))


if __name__ == '__main__':
    unittest.main()