# Data Preprocessing
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_sentences_and_labels(file_name):
    file = open(file_name,'r')
    sentence1 = []
    sentence2 = []
    labels = []
    for line in file :
        data = json.loads(line)
        labels.append(data['gold_label'])
        s1 = data['sentence1_binary_parse']
        s2 = data['sentence2_binary_parse']
        s1 = s1.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()
        s2 = s2.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()
        sentence1.append(' '.join(s1))
        sentence2.append(' '.join(s2))


    return sentence1,sentence2,labels

train_data = get_sentences_and_labels('snli_1.0_train.jsonl')
test_data = get_sentences_and_labels('snli_1.0_test.jsonl')
validation_data = get_sentences_and_labels('snli_1.0_dev.jsonl')




# Tokenization and vector representation of all the words.
file = open('glove.6B.200d.txt',encoding = 'utf-8')
embedding_dict = {}
for line in file:
    values = line.split()
    word = values[0]
    embedding_dict[word] = np.array(values[1:])


def labels(names):
    labels = []
    for ele in names :
        if ele == 'entailment':
            labels.append(0)
        elif ele == 'contradiction':
            labels.append(1)
        else:
            labels.append(2)
    return labels
train_labels = labels(train_data[2])
validation_labels = labels(validation_data[2])
test_labels = labels(test_data[2])


num_words = None
OOV_TOKEN = '<OOV>'
padding_type = 'pre'
truncating_type = 'pre'
maxlength = 25
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = num_words,oov_token = OOV_TOKEN)
tokenizer.fit_on_texts(train_data[0] + train_data[1])
train_1 = tokenizer.texts_to_sequences(train_data[0])
train_2 = tokenizer.texts_to_sequences(train_data[1])
train_1 = pad_sequences(train_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
train_2 = pad_sequences(train_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
train_sequences = []
for i in range(550152):
    train_sequences.append(np.concatenate((train_1[i],train_2[i])))

validation_1 = tokenizer.texts_to_sequences(validation_data[0])
validation_2 = tokenizer.texts_to_sequences(validation_data[1])
validation_1 = pad_sequences(validation_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
validation_2 = pad_sequences(validation_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
validation_sequences = []
for i in range(10000):
    validation_sequences.append(np.concatenate((validation_1[i] , validation_2[i])))

test_sequences = []
test_1 = tokenizer.texts_to_sequences(test_data[0])
test_2 = tokenizer.texts_to_sequences(test_data[1])
test_1 = pad_sequences(test_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
test_2 = pad_sequences(test_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
for i in range(10000):
    test_sequences.append(np.concatenate((test_1[i], test_2[i])))





''' Implementation of the Attention layer '''
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers,regularizers,constraints
def dot_product(x, kernel):
  if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
  else:
        return K.dot(x, kernel)
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(AttentionWithContext,self).get_config()


''' End of the attention layer'''



embedding_matrix = np.zeros((33297,200),dtype = 'float')

count = 0

for word,index in tokenizer.word_index.items():
    try:
        vector = embedding_dict[word]
        embedding_matrix[index] = vector
    except:
        count += 1


#print(count)

from tensorflow.keras.layers import Input
vocab_size = 33296


inp = Input(shape = (50,))
from tensorflow.keras.models import Model
x = tf.keras.layers.Embedding(vocab_size + 1, 200,input_length = 50,weights = [embedding_matrix],trainable = False)(inp)
x = (tf.keras.layers.Dropout(0.1))(x)
x = (tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences = True)))(x)
x = (tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences = True)))(x)
x = (tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences = True)))(x)
x = AttentionWithContext()(x)
x = (tf.keras.layers.Dense(32,activation = 'relu'))(x)
x = (tf.keras.layers.Dense(16,activation = 'relu'))(x)
x = (tf.keras.layers.Dense(3,activation = 'softmax'))(x)

model_glove = Model(inputs = inp,outputs = x)
model_glove.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
#model_glove.summary()

history = model_glove.fit(np.array(train_sequences),np.array(train_labels),epochs = 10,validation_data = (np.array(validation_sequences),np.array(validation_labels)))

'''
model_glove.evaluate(np.array(validation_sequences),np.array(validation_labels))
model_glove.evaluate(np.array(test_sequences),np.array(test_labels))
model_glove.evaluate(np.array(train_sequences),np.array(train_labels))

train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
validation_loss = history.history['val_loss']

'''

# Saving the model
model_json = model_glove.to_json()
with open("Glove_parameters.json",'w') as json_file:
  json_file.write(model_json)

model_glove.save_weights("Glove_weights.h5")
