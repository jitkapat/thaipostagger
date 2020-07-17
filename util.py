import tensorflow as tf
import tokenization as tok
import re
import numpy as np
from string import punctuation
from pythainlp import thai_digits
from tensorflow.keras.layers import concatenate
from transformers import TFBertModel

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def load_bert():
    bert = TFBertModel.from_pretrained('bert_base_th')
    return bert

def load_tokenizer():
    tf.gfile = tf.io.gfile
    tokenizer = tok.ThaiTokenizer('th.wiki.bpe.op25000.vocab', 'th.wiki.bpe.op25000.model')
    return tokenizer

def build_model(max_seq_length):
    n_tags = 14
    bert = load_bert()
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype='int64')
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks", dtype='int64')
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids", dtype='int64')
    orthog_in = tf.keras.layers.Input(shape=(max_seq_length, 6))
    bert_inputs = [in_id, in_mask, in_segment]
    all_inputs = bert_inputs + [orthog_in]
    bert_output = bert(bert_inputs)[0]
    x = concatenate([bert_output, orthog_in], axis=-1)
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)
    out = tf.keras.layers.Dense(n_tags, activation=tf.keras.activations.softmax)(x)
    model = tf.keras.models.Model(inputs=all_inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00004), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])   
    return model

def load_model():
    model = build_model(MAX_SEQUENCE_LENGTH+2) 
    model.load_weights('bert_base_th/AACL_BERT_TH.hdf5')
    return model

def convert_examples_to_features(tokenizer, examples, max_seq_length=110):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids = [], [], []
    for example in examples:
        input_id, input_mask, segment_id = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        
    )

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        return input_ids, input_mask, segment_ids
    
    tokens_a = example.text_a
    if len(tokens_a) > max_seq_length-2:
        tokens_a = tokens_a[0 : (max_seq_length-2)]

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]   
    orig_to_tok_map = []              
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)
    #print(len(tokens_a))
    for token in tokens_a:
        orig_to_tok_map.append(len(tokens))       
        tokens.extend(tokenizer.tokenize(token))
        #orig_to_tok_map.append(len(tokens)-1)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)
    input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])
    #print(len(orig_to_tok_map), len(tokens), len(input_ids), len(segment_ids)) #for debugging

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    #print(len(label_ids)) #for debugging
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids

def convert_text_to_examples(texts):
    """Create InputExamples"""
    InputExamples = []
    for text in texts:
        InputExamples.append(
            InputExample(guid=None, text_a=text, text_b=None)
        )
    return InputExamples

def is_thai_alpha(word):
    return int(bool(re.match(r'^[ก-๏\s]+$', word)))

def is_roman_alpha(word):
    return int(bool(re.match(r'^[a-zA-Z\s]+$', word)))

def is_alnum(word):
    return int(bool(re.match(r'^[ก-๙a-zA-Z0-9\s]+$', word)))

def is_capitalized(word):
    return int(bool(re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', word)))

def is_punct(word):
    return int(bool(re.match(r'^[{}\s]+$'.format(punctuation), word)))      

def is_num(word):
    return int(bool(re.match(r'^[๐-๙0-9]+$', word)))
    
def orthog(word):
    return {
        '1is_roman': is_roman_alpha(word),
        '2is_thai': is_thai_alpha(word),
        '3all_digit': is_num(word),
        '4has_digit': int(any([char in word for char in all_thai_digits])),
        '5all_punct': is_punct(word),
        '6has_punct': int(any([char in word for char in punctuation])),
    }

def orthog_to_vector(feature_dict):
    feature_vec = np.zeros(len(feature_dict))
    for i, k in enumerate(sorted(feature_dict.keys())):
        feature_vec[i] = feature_dict[k]
    return feature_vec

def all_orthog_vec(X):
    vec = np.zeros((len(X), MAX_SEQUENCE_LENGTH+2, 6))
    for i, seq in enumerate(X):
        for j, word in enumerate(seq):
            feature = orthog(word)
            for k, feat in enumerate(sorted(feature.keys())):
                vec[i][j+1][k] = feature[feat]
    return vec

def preprocess_one_text(text):
    """
    text = a list of word tokens
    space is necessary to remove because it can interfere with the tokenizer.
    """
    return [word.replace(' ', '_') for word in text]

def preprocess_text_list(text_list):
    return [preprocess_one_text(text) for text in text_list]

def text_list_to_feature(text_list, tokenizer, max_seq_length): 
    """
    text_list = list of tokenized, preprocessed text samples 
    e.g. [['สวัสดี', 'ครับ'], ['สวัสดี, 'ค่ะ']]
    """
    examples = convert_text_to_examples(text_list)
    input_ids, input_masks, segment_ids = convert_examples_to_features(tokenizer, examples, max_seq_length=MAX_SEQUENCE_LENGTH+2)
    orthog = all_orthog_vec(text_list)
    return input_ids, input_masks, segment_ids, orthog

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 110
    punctuation += 'ฯ'
    punctuation += ' '
    all_thai_digits = [num for num in thai_digits]+ '1 2 3 4 5 6 7 8 9 0'.split(' ')