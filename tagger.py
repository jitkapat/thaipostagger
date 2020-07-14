class Tagger:

    __instance = None

    model_path = './model/taggerv0.1.bin'

    @staticmethod
    def get_instance():
        if Tagger.__instance is None:
            Tagger()
        return Tagger.__instance

    def __init__(self):
        if Tagger.__instance != None:
            raise Exception("This class is a singleton!")
        self.model = None # Need to implement model loading
        Tagger.__instance = self
    
    def tag(self, tokens:list):
        """ Tag tokens with POS tags

        Arg:
            tokens: a list of Thai tokens

        Returns:
            a list of POS tags of the same length
        """
        if type(tokens) != list:
            raise TypeError('The input must be a list of strings')
        return ['NOUN' for x in range(len(tokens))]
    