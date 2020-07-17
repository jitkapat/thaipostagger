from util import *

class Tagger:

    __instance = None

    @staticmethod
    def get_instance():
        if Tagger.__instance is None:
            Tagger()
        return Tagger.__instance

    def __init__(self):
        if Tagger.__instance != None:
            raise Exception("This class is a singleton!")
        Tagger.__instance = self
        self.max_seq_length = 110
        self.model = load_model(self.max_seq_length)
        self.tokenizer = load_tokenizer()
        self.int2tag = {0: '-PAD-',
                        1: 'ADP',
                        2: 'ADV',
                        3: 'AUX',
                        4: 'CCONJ',
                        5: 'DET',
                        6: 'NOUN',
                        7: 'NUM',
                        8: 'PART',
                        9: 'PRON',
                        10: 'PROPN',
                        11: 'PUNCT',
                        12: 'SCONJ',
                        13: 'VERB'}
    
    def tag(self, text:list):
        """ Tag tokens with POS tags

        Arg:
            text: a list of Thai tokens or list of list of Thai tokens

        Returns:
            a list of POS tags of the same length or a list of such lists
        """
        if type(text) != list:
            raise TypeError('The input must be a list of strings or list of lists of string')
        if all(type(elem)==str for elem in text):
            text = [text]
        preprocessed_text = preprocess_text_list(text)
        features = text_list_to_feature(preprocessed_text, self.tokenizer, self.max_seq_length)
        predictions = self.model.predict(features).argmax(axis=-1)
        tags = [[self.int2tag[tag] for tag in seq[1:len(text[i])+1]] for i, seq in enumerate(predictions)]
        if len(tags) == 1:
            return tags[0]
        return tags
    