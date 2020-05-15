# Text Preprocessing
import json
from sklearn.feature_extraction.text import TfidfVectorizer

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





vectorizer = TfidfVectorizer(stop_words = None ,max_features = 25000)
vectorizer.fit(train_data[0] + train_data[1])

tfidf_train_1 = vectorizer.transform(train_data[0])
tfidf_train_2 = vectorizer.transform(train_data[1])

tfidf_test_1 = vectorizer.transform(test_data[0])
tfidf_test_2 = vectorizer.transform(test_data[1])

tfidf_validation_1 = vectorizer.transform(validation_data[0])
tfidf_validation_2 = vectorizer.transform(validation_data[1])





from scipy import sparse
import numpy as np
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






from scipy import sparse
import tensorflow as tf
def labels(names):
    num_labels = []
    for ele in names:
        if ele == 'entailment':
            num_labels.append(0)
        elif ele == 'neutral':
            num_labels.append(1)
        else:
            num_labels.append(2)
    return num_labels

train_labels = labels(train_data[2])
validation_labels = labels(validation_data[2])
test_labels = labels(test_data[2])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,activation = 'relu',input_shape = [50000]),
    tf.keras.layers.Dense(16,activation = 'relu'),
    tf.keras.layers.Dense(3,activation = 'softmax')
])
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.summary()




# here with epochs = 40, we will get very high train_accuracy of 0.9922 as mentioned in the table. 
model.fit(train_final,np.array(train_labels),epochs = 5,validation_data = (validation_final,np.array(validation_labels)))
val_acc = model.evaluate(validation_final,np.array(validation_labels))
test_acc = model.evaluate(test_final,np.array(test_labels))
train_acc = model.evaluate(train_final,np.array(train_labels))


model_json = model.to_json()
with open('LR_implemented_parameters.json','w') as json_file:
  json_file.write(model_json)

model.save_weights('LR_implemented_weights.h5')
