# I have commented out all the parts that uses the validation set. If you want to run on validation set please uncommment correspondingly.
# text preprocessing. THis is needed for all the models
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

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
#validation_data = get_sentences_and_labels('snli_1.0_dev.jsonl')

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
#validation_labels = labels(validation_data[2])
test_labels = labels(test_data[2])




''' sklearn  Logistic regression '''
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = None, max_features = 25000)
vectorizer.fit(train_data[0] + train_data[1])

train_1 = vectorizer.transform(train_data[0])
train_2 = vectorizer.transform(train_data[1])
tfidf_train = sparse.hstack((train_1,train_2),format = 'csr')

'''
validation_1 = vectorizer.transform(validation_data[0])
validation_2 = vectorizer.transform(validation_data[1])
tfidf_validation = sparse.hstack((validation_1,validation_2),format = 'csr')
'''

test_1 = vectorizer.transform(test_data[0])
test_2 = vectorizer.transform(test_data[1])
tfidf_test = sparse.hstack((test_1,test_2),format = 'csr')


tfidf_sklearn_model = joblib.load('LR_sklearn_parameters.sav')

train_acc = tfidf_sklearn_model.score(tfidf_train,train_data[2])
train_predictions  = tfidf_sklearn_model.predict(tfidf_train)

'''
val_acc = tfidf_sklearn_model.score(tfidf_validation,validation_data[2])
val_predictions = tfidf_sklearn_model.predict(tfidf_validation)
'''

test_acc = tfidf_sklearn_model.score(tfidf_test,test_data[2])
test_predictions = tfidf_sklearn_model.predict(tfidf_test)

file_tfidf = open("tfidf.txt",'w')
for ele in test_predictions:
    file_tfidf.write(ele)
    file_tfidf.write('\n')
file_tfidf.close()

#print(train_acc,val_acc,test_acc)



''' End of Sklearn Logistic Regression model '''





''' Manually implemented Logistic regression.
    Evaluation of the model is kept in comments as
    this is not the model needed for the project '''

'''
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
vectorizer = TfidfVectorizer(stop_words = None,max_features = 25000)
vectorizer.fit(train_data[0] + train_data[1])

tfidf_train_1 = vectorizer.transform(train_data[0])
tfidf_train_2 = vectorizer.transform(train_data[1])

tfidf_test_1 = vectorizer.transform(test_data[0])
tfidf_test_2 = vectorizer.transform(test_data[1])

tfidf_validation_1 = vectorizer.transform(validation_data[0])
tfidf_validation_2 = vectorizer.transform(validation_data[1])

def final(first,second,N):
    final = sparse.lil_matrix((N,50000),dtype = float)
    for i in range(N):
        final[i] = sparse.hstack((first[i],second[i]),format = 'lil')
    return final


train_final = final(tfidf_train_1,tfidf_train_2,550152)
validation_final = final(tfidf_validation_1,tfidf_validation_2,10000)
test_final = final(tfidf_test_1,tfidf_test_2,10000)
#print(tfidf_train_1.shape,tfidf_train_2.shape,tfidf_test_1.shape,tfidf_test_2.shape)
#print(train_final.shape,validation_final.shape,test_final.shape)

LR_implemented_json_file = open('LR_implemented_parameters.json','r')
LR_implemented_json = LR_implemented_json_file.read()
LR_implemented_json_file.close()

LR_implemented_model = tf.keras.models.model_from_json(LR_implemented_json,custom_objects = None)
LR_implemented_model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

train_acc = LR_implemented_model.evaluate(train_final,np.array(train_labels))
val_acc = LR_implemented_model.evaluate(validation_final,np.array(validation_labels))
test_acc = LR_implemented_model.evaluate(test_final,np.array(test_labels))

print(train_acc,val_acc,test_acc)

'''

''' End of the manually implemented Logistic regression '''








''' Word embeddings model.
    Evaluation of this model is also in comments as
    this is not the proposed model for the project '''
'''

#Preprocessing
num_words = 27000
OOV_TOKEN = '<OOV>'
padding_type = 'pre'
truncating_type = 'pre'
maxlength = 20
tokenizer = Tokenizer(oov_token = OOV_TOKEN,num_words = num_words )
tokenizer.fit_on_texts(train_data[0] + train_data[1])
train_1 = tokenizer.texts_to_sequences(train_data[0])
train_2 = tokenizer.texts_to_sequences(train_data[1])
train_1 = pad_sequences(train_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
train_2 = pad_sequences(train_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

train_sequences = []
for i in range(550152):
    train_sequences.append(np.concatenate((train_1[i] , train_2[i])))

validation_1 = tokenizer.texts_to_sequences(validation_data[0])
validation_2 = tokenizer.texts_to_sequences(validation_data[1])
validation_1 = pad_sequences(validation_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
validation_2 = pad_sequences(validation_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

validation_sequences = []
for i in range(10000):
    validation_sequences.append(np.concatenate((validation_1[i],validation_2[i])))


test_1 = tokenizer.texts_to_sequences(test_data[0])
test_2 = tokenizer.texts_to_sequences(test_data[1])
test_1 = pad_sequences(test_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
test_2 = pad_sequences(test_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

test_sequences = []
for i in range(10000):
    test_sequences.append(np.concatenate((test_1[i] , test_2[i])))

# Attention layer implementation

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


# end of the implementation of the attention layer

embedding_json_file = open("word_embeddings_parameters.json",'r')
embedding_json_model = embedding_json_file.read()
embedding_model = tf.keras.models.model_from_json(embedding_json_model,custom_objects = {'AttentionWithContext':AttentionWithContext()})
embedding_model.load_weights('word_embeddings_weights.h5')
embedding_model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
embedding_model.evaluate(np.array(train_sequences),np.array(train_labels))
embedding_model.evaluate(np.array(validation_sequences),np.array(validation_labels))
embedding_model.evaluate(np.array(test_sequences),np.array(test_labels))

'''



''' Glove embeddings model '''

# Preprocessing for the Glove model.
num_words = None
OOV_TOKEN = '<OOV>'
padding_type = 'pre'
truncating_type = 'pre'
maxlength = 25
tokenizer = Tokenizer(oov_token = OOV_TOKEN,num_words = num_words )
tokenizer.fit_on_texts(train_data[0] + train_data[1])
train_1 = tokenizer.texts_to_sequences(train_data[0])
train_2 = tokenizer.texts_to_sequences(train_data[1])
train_1 = pad_sequences(train_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
train_2 = pad_sequences(train_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

train_sequences = []
for i in range(550152):
    train_sequences.append(np.concatenate((train_1[i] , train_2[i])))
'''
validation_1 = tokenizer.texts_to_sequences(validation_data[0])
validation_2 = tokenizer.texts_to_sequences(validation_data[1])
validation_1 = pad_sequences(validation_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
validation_2 = pad_sequences(validation_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

validation_sequences = []
for i in range(10000):
    validation_sequences.append(np.concatenate((validation_1[i],validation_2[i])))
'''

test_1 = tokenizer.texts_to_sequences(test_data[0])
test_2 = tokenizer.texts_to_sequences(test_data[1])
test_1 = pad_sequences(test_1,maxlen = maxlength,padding = padding_type,truncating = truncating_type)
test_2 = pad_sequences(test_2,maxlen = maxlength,padding = padding_type,truncating = truncating_type)

test_sequences = []
for i in range(10000):
    test_sequences.append(np.concatenate((test_1[i] , test_2[i])))



'''  Attention layer implementation '''

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


''' end of the implementation of the attention layer '''

glove_json = open("Glove_parameters.json",'r')
glove_model_json = glove_json.read()
glove_json.close()
glove_model = tf.keras.models.model_from_json(glove_model_json,custom_objects = {'AttentionWithContext': AttentionWithContext()})
glove_model.load_weights("Glove_weights.h5")
glove_model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
#val_acc,val_loss = glove_model.evaluate(np.array(validation_sequences),np.array(validation_labels))
test_acc,test_loss = glove_model.evaluate(np.array(test_sequences),np.array(test_labels))
#validation_predictions = glove_model.predict(np.array(validation_sequences))
test_predictions = glove_model.predict(np.array(test_sequences))

''' End of the glove embeddings model '''

file_deep = open('deep_model.txt','w')

for el in test_predictions:
    ele = np.argmax(el)
    if ele == 0:
        file_deep.write('entailment\n')
    elif ele == 1:
        file_deep.write('contradiction\n')
    else:
        file_deep.write('neutral\n')

file_deep.close()
