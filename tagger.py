import util


class Tagger:

    def __init__(self):
        self.tagger = TaggerSingleton.get_instance()

    def tag(self, text):
        return self.tagger.tag(text)


class TaggerSingleton:

    __instance = None

    @staticmethod
    def get_instance():
        if TaggerSingleton.__instance is None:
            TaggerSingleton()
        return TaggerSingleton.__instance

    def __init__(self):
        if TaggerSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        TaggerSingleton.__instance = self
        self.max_seq_length = 110
        self.model = None
        self.tokenizer = None
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
            text: a single Thai token (str), a list of Thai tokens, or list of list of Thai tokens.

        Returns:
            A pos tag of the input token in case of a single token.
            Otherwise a list of POS tags of the same length as input, or a list of such lists.
        """
        
        if not text:
            raise TypeError('The input is empty')
        if type(text) != list:
            if type(text) == str:
                text = [text]
            else:
                raise TypeError('The input must be a string, list of strings or list of lists of string')
        if not all(type(elem)==str for elem in text): # check if input is list of str or list of lists of str
            if not all(type(elem)==list for elem in text):
                raise TypeError('The input is a list of neither list or string')
            elif not(all(all(type(subelem)==str for subelem in elem) for elem in text)):
                raise TypeError('The input is a list of list of not string')
        elif all(type(elem)==str for elem in text):
            text = [text]

        preprocessed_text = util.preprocess_text_list(text)
        features = util.text_list_to_feature(preprocessed_text, self.tokenizer, self.max_seq_length)
        predictions = self.model.predict(features).argmax(axis=-1)
        tags = [[self.int2tag[tag] for tag in seq[1:len(text[i])+1]] for i, seq in enumerate(predictions)]
        if len(tags) == 1:
            return tags[0]
        return tags
    
def pos_tag(text):
    postagger = Tagger()
    return postagger.tag(text)
