import tensorflow as tf
import tokenizer as tok
from tensorflow.keras.layers import concatenate
from transformers import TFBertModel

def bert():
    bert = TFBertModel.from_pretrained('bert_base_th')
    return bert

def tokenizer():
    tokenizer = tok.ThaiTokenizer('th.wiki.bpe.op25000.vocab', 'th.wiki.bpe.op25000.model')
    return tokenizer

def build_model(max_seq_length):
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
    model.summary()
    return model

def load_model():
    model = build_model(MAX_SEQUENCE_LENGTH+2) 
    model.load_weights('AACL_BERT_TH.hdf5')
    return model

if __name__ == "__main__":
    n_tags = 14
    MAX_SEQUENCE_LENGTH = 110
    tf.gfile = tf.io.gfile
    bert = load_bert