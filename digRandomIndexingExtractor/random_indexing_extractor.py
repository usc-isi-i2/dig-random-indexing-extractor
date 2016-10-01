from digExtractor.extractor import Extractor
import re
import copy 
import types
import sys
from itertools import ifilter
from nltk import sent_tokenize, word_tokenize, data
from sklearn.preprocessing import normalize

class RandomIndexingExtractor(Extractor):

    def __init__(self):
        self.renamed_input_fields = ['tokens', 'values']
        try:
            mypath = data.find("tokenizers/punkt")
        except LookupError as e: 
            print "Unable to find nltk tokenizers/punkt.  Please install it!"
            print e

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

    def get_feature_model(self):
        return self.feature_model

    def set_feature_model(self, feature_model):
        self.feature_model = feature_model
        return self

    def tokenize_string(self, string):
        """
        I designed this method to be used independently of an obj/field. If this is the case, call _tokenize_field.
        It's more robust.
        :param string: e.g. 'salt lake city'
        :return: list of tokens
        """
        list_of_sentences = list()
        tmp = list()
        tmp.append(string)
        k = list()
        k.append(tmp)
        # print k
        list_of_sentences += k  # we are assuming this is a unicode/string

        word_tokens = list()
        for sentences in list_of_sentences:
            # print sentences
            for sentence in sentences:
                for s in sent_tokenize(sentence):
                    word_tokens += word_tokenize(s)

        return word_tokens

    def symmetric_vector_generator(self, word, list_of_words, embeddings_dict, window_size=2, multi=False):
        """
        The algorithm will search for occurrences of word in list_of_words (there could be multiple or even 0), then
        symmetrically look backward and forward up to the window_size. If the words in the window (not including
        the word itself) are in the embeddings_dict, we will add them up, and that constitutes a context_vec.
        If the word is not there in embeddings_dict, we do not include it. Note that if no words in the embeddings_dict
        then we will act as if the word itself had never occurred in the list_of_words.
        :param word:
        :param list_of_words: e.g. high_recall_readability_text
        :param embeddings_dict:
        :param window_size
        :param multi: If True, then word is multi-token. You must tokenize it first, then generate context embedd.
        :return: a list of lists, with each inner list representing the context vectors. If there are no occurrences
        of word, will return None. Check for this in your code.
        """
        if not list_of_words:
            return None
        context_vecs = list()
        if multi:
            word_tokens = self.tokenize_string(word)
        for i in range(0, len(list_of_words)):
            if multi:
                if list_of_words[i] == word_tokens[0] and list_of_words[i:i + len(word_tokens)] == word_tokens:
                    min_index = i-window_size
                    max_index = ((i + len(word_tokens))-1)+window_size
                else:
                    continue
            elif list_of_words[i] != word:
                continue
            else:
                min_index = i-window_size
                max_index = i+window_size

            # make sure the indices are within range
            if min_index < 0:
                min_index = 0
            if max_index >= len(list_of_words):
                max_index = len(list_of_words)-1

            new_context_vec = []
            for j in range(min_index, max_index+1):
                if multi:   # we do not want the vector of the word/work_tokens itself
                    if j >= i and j < i+len(word_tokens):
                        continue
                elif j == i:
                    continue

                if list_of_words[j] not in embeddings_dict: # is the word even in our embeddings?
                    continue

                vec = list(embeddings_dict[list_of_words[j]])  # deep copy of list
                if not new_context_vec:
                    new_context_vec = vec
                else:
                    self._add_vectors(new_context_vec, vec)
            if not new_context_vec:
                continue
            else:
                context_vecs.append(new_context_vec)
        if not context_vecs:
            return None
        else:
            return context_vecs

    def is_sublist_in_big_list(self, big_list, sublist):
        # matches = []
        for i in range(len(big_list)):
            if big_list[i] == sublist[0] and big_list[i:i + len(sublist)] == sublist:
                return True
        return False

    def _add_vectors(self, big_vector, little_vector):
        """
        little_vector gets added into big_vector. Vectors must be same length. big_vector may get modified.
        :param big_vector:
        :param little_vector:
        :return: None
        """
        if len(little_vector) != len(big_vector):
            raise Exception('Error! Vector lengths are different!')

        else:
            for i in range(0, len(little_vector)):
                big_vector[i] += little_vector[i]

    def get_feature_vector(self, tokens, value):
        # feature_vector is a misnomer here, as there is a matrix of vectors per word.
        # But we deal with that when predicting.

        lower_tokens = [token.lower() for token in tokens]
        lower_value = value.lower()
        word_tokens = self.tokenize_string(lower_value)
        if len(word_tokens) <= 1:  # we're dealing with a single-token word
            if lower_value not in lower_tokens:
                return None
            context_vecs = self.symmetric_vector_generator(lower_value, lower_tokens, self.embeddings)
        elif self.is_sublist_in_big_list(lower_tokens, word_tokens):
            context_vecs = self.symmetric_vector_generator(lower_value, lower_tokens, self.embeddings, multi=True)
        else:
            return None

        if context_vecs:
            context_vecs = normalize(context_vecs)
            return context_vecs
        else:
            return None

    def get_model_prediction(self, feature_vector):
        # we'll be conservative. If any of the context vecs return True, classify the word as relevant.
        model = self.get_model()
        feature_model = self.get_feature_model()
        feature_vector = feature_model.transform(feature_vector)
        predicted_labels = model.predict(feature_vector)
        for label in predicted_labels:
            if label == 1:
                return True
        return False

    def predict(self, tokens, value):
        feature_vector = self.get_feature_vector(tokens, value)
        # print feature_vector
        if feature_vector is not None:
            return self.get_model_prediction(feature_vector)
        else:
            return False

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

