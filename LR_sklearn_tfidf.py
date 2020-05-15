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




vectorizer = TfidfVectorizer(stop_words = None, max_features = 25000)
vectorizer.fit(train_data[0] + train_data[1])

tfidf_train_1 = vectorizer.transform(train_data[0])
tfidf_train_2 = vectorizer.transform(train_data[1])

tfidf_test_1 = vectorizer.transform(test_data[0])
tfidf_test_2 = vectorizer.transform(test_data[1])

tfidf_validation_1 = vectorizer.transform(validation_data[0])
tfidf_validation_2 = vectorizer.transform(validation_data[1])

tfidf_train = sparse.hstack((tfidf_train_1,tfidf_train_2),format = 'csr')
tfidf_validation = sparse.hstack((tfidf_validation_1,tfidf_validation_2),format = 'csr')
tfidf_test = sparse.hstack((tfidf_test_1,tfidf_test_2),format = 'csr')





epochs = 1000
from scipy import sparse
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
    intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=epochs,
    multi_class='auto', verbose=2, warm_start=False, n_jobs=None, l1_ratio=None

)

history = model.fit(tfidf_train, train_data[2])




import joblib
joblib.dump(model,'LR_sklearn_parameters.sav')




'''
# Final accuracies obtained.

train_acc = model.score(tfidf_train,train_data[2])
print("train accuracy is ",train_acc)

val_acc = model.score(tfidf_validation,validation_data[2])
print("Validation accuracy is ",val_acc)

test_acc = model.score(tfidf_test,test_data[2])
print("test accuracy is ",test_acc)
'''
