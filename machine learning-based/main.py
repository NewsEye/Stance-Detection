import csv
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from collections import defaultdict
from test import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier   
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost
#Gradient Boosting Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation

classifiers=[
    
    (LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=100.0, fit_intercept=True,
    intercept_scaling=10, class_weight=None, random_state=None, solver='warn', max_iter=10,
    multi_class='warn', verbose=0, warm_start=False, n_jobs=None),"Logistic Regression"),
    
    
    (KNeighborsClassifier(1),"K Nearest Classifier "),
    
    (SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False),'Support Vector Machine Classifier'),
    
    (QuadraticDiscriminantAnalysis(),'Qudratic Discriminant Analysis'),
    
    (RandomForestClassifier(max_depth=50, n_estimators=10, max_features=1),'Random Forest Classifier'),
    
    (AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=0.01,
                        algorithm='SAMME.R', random_state=None),'Adaboost Classifier'),
    (SGDClassifier(),'SGD Classifier'),
    
    (DecisionTreeClassifier(max_depth=5),'Decision Tree Classifier'),
    (xgboost.XGBClassifier(learning_rate=0.1),'XG Boost Classifier'),
    
    (LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, 
        store_covariance=False,tol=0.00001),'Linear Discriminant Analysis'),
     
    (GaussianNB(),'Gaussian Naive Bayes ')
]

csv.field_size_limit(sys.maxsize)

f_bodies = open('data/train_puls_body_texts.csv', 'r', encoding='utf-8')
csv_bodies = csv.DictReader(f_bodies)
bodies = []
for row in csv_bodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(bodies):
        bodies += [None] * (body_id + 1 - len(bodies))
    bodies[body_id] = row['articleBody']
#    print(row)
f_bodies.close()
body_inverse_index = {bodies[i]: i for i in range(len(bodies))}

all_pos, all_neg, all_obj = [], [], []  # each article = (target, body, stance)

f_stances = open('data/train_puls_stance.csv', 'r', encoding='utf-8')
csv_stances = csv.DictReader(f_stances)
for row in csv_stances:
    body = bodies[int(row['Body ID'])]
    if row['Stance'] == 'POS':
        all_pos.append((row['target'], body, row['Stance']))
    elif row['Stance'] == 'NEG':
        all_neg.append((row['target'], body, row['Stance']))
    elif row['Stance'] == 'OBJ':
        all_obj.append((row['target'], body, row['Stance']))
#    elif row['Stance'] == 'disagree':
#        all_disagree.append((row['target'], body, row['Stance']))
#    print(row)
f_stances.close()

print('\tPOS\tNEG\tOBJ')
print('All\t', len(all_pos), '\t', len(all_neg), '\t', len(all_obj))

train_pos = all_pos[:len(all_pos) * 9 // 10]
train_neg = all_neg[:len(all_neg) * 9 // 10]
train_obj = all_obj[:len(all_obj) * 9 // 10]

val_pos = all_pos[len(all_pos) * 9 // 10:]
val_neg = all_neg[len(all_neg) * 9 // 10:]
val_obj = all_obj[len(all_obj) * 9 // 10:]

train_pos = all_pos[:len(all_pos) //100]
train_neg = all_neg[:len(all_neg) //100]
train_obj = all_obj[:len(all_obj) //100]

val_pos = all_pos[len(all_pos) * 9 // 10:]
val_neg = all_neg[len(all_neg) * 9 // 10:]
val_obj = all_obj[len(all_obj) * 9 // 10:]

val_pos = val_pos[len(val_pos) * 9 // 10:]
val_neg = val_neg[len(val_neg) * 9 // 10:]
val_obj = val_obj[len(val_obj) * 9 // 10:]

print('Train\t', len(train_pos), '\t', len(train_neg), '\t', len(train_obj))
print('Valid.\t', len(val_pos), '\t', len(val_neg), '\t', len(val_obj))

train_all = (train_pos + train_neg + train_obj)
# each article = (target, body, stance)
random.Random(0).shuffle(train_all)
train_all = np.array(train_all)

val_all = val_pos + val_neg + val_obj
random.Random(0).shuffle(val_all)
val_all = np.array(val_all)

# Tokenise text
pattern = re.compile("[^a-zA-Z0-9 ]+")  # strip punctuation, symbols, etc.
stop_words = set(stopwords.words('english'))
def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text

# Compute term-frequency of words in documents
def doc_to_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram+1):
            if i - j < 0:
                break
            word = [words[i-k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret

# Build corpus of article bodies and targets in training dataset
corpus = np.r_[train_all[:, 1], train_all[:, 0]]

# Learn idf of every word in the corpus
df = defaultdict(float)
for doc in tqdm(corpus):
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df[word] += 1.0
            seen.add(word)

num_docs = corpus.shape[0]
idf = defaultdict(float)
for word, val in tqdm(df.items()):
    idf[word] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0  # smoothed idf

# Load embedding word vectors
f_embed = open("data_old/embeddings/cc.en.300.vec", "rb")  
embed_vectors = {}
for line in tqdm(f_embed):
    embed_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

# Convert a document to embed vectors, by computing tf-idf of each word * embed of word / total tf-idf for document
def doc_to_embed(doc):
    doc_tf = doc_to_tf(doc)
    doc_tf_idf = defaultdict(float)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]
        
    doc_vector = np.zeros(embed_vectors['embed'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
        return doc_vector
    
    for word, tf_idf in doc_tf_idf.items():
        if word in embed_vectors:
            doc_vector += embed_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector

# Compute cosine similarity of Embed vectors for all target-body pairs
def dot_product(vec1, vec2):
    sigma = 0.0
    for i in range(vec1.shape[0]):  # assume vec1 and vec2 has same shape
        sigma += vec1[i] * vec2[i]
    return sigma
    
def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))
        
def cosine_similarity(doc):
    target_vector = doc_to_embed(doc[0])
    body_vector = doc_to_embed(doc[1])
    
    if magnitude(target_vector) == 0.0 or magnitude(body_vector) == 0.0:  # edge case: document is empty
        return 0.0
    
    return dot_product(target_vector, body_vector) / (magnitude(target_vector) * magnitude(body_vector))


# Compute the KL-Divergence of language model (LM) representations of the target and the body
def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma

def kl_divergence(doc, eps=0.1):
    # Convert target and body to 1-gram representations
    tf_target = doc_to_tf(doc[0])
    tf_body = doc_to_tf(doc[1])
    
    # Convert dictionary tf representations to vectors (make sure columns match to the same word)
    words = set(tf_target.keys()).union(set(tf_body.keys()))
    vec_target, vec_body = np.zeros(len(words)), np.zeros(len(words))
    i = 0
    for word in words:
        vec_target[i] += tf_target[word]
        vec_body[i] = tf_body[word]
        i += 1
    
    # Compute a simple 1-gram language model of target and body
    lm_target = vec_target + eps
    lm_target /= np.sum(lm_target)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)
    
    # Return KL-divergence of both language models
    return divergence(lm_target, lm_body)


# Compute the KL-Divergence of language model (LM) representations of the target and the body
def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma

def kl_divergence(doc, eps=0.1):
    # Convert target and body to 1-gram representations
    tf_target = doc_to_tf(doc[0])
    tf_body = doc_to_tf(doc[1])
    
    # Convert dictionary tf representations to vectors (make sure columns match to the same word)
    words = set(tf_target.keys()).union(set(tf_body.keys()))
    vec_target, vec_body = np.zeros(len(words)), np.zeros(len(words))
    i = 0
    for word in words:
        vec_target[i] += tf_target[word]
        vec_body[i] = tf_body[word]
        i += 1
    
    # Compute a simple 1-gram language model of target and body
    lm_target = vec_target + eps
    lm_target /= np.sum(lm_target)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)
    
    # Return KL-divergence of both language models
    return divergence(lm_target, lm_body)


# Other feature 1
def ngram_overlap(doc):
    # Returns how many times n-grams (up to 3-gram) that occur in the article's target occur on the article's body.
    tf_target = doc_to_tf(doc[0], ngram=3)
    tf_body = doc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_target.keys():
        if words in tf_body:
            matches += tf_body[words]
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  # normalise for document length

# Define function to convert (target, body text) to feature vectors for each document
ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec

# Initialise X (matrix of feature vectors) for train dataset
x_train = np.array([to_feature_array(doc) for doc in tqdm(train_all)])

# Define label <-> int mappings for y
label_to_int = {'POS': 0, 'NEG': 1, 'OBJ': 2}
int_to_label = ['POS', 'NEG', 'OBJ']

# Initialise Y (gold output vector) for train dataset
y_train = np.array([label_to_int[i] for i in train_all[:, 2]])

# Initialise x (feature vectors) for validation dataset
x_val = np.array([to_feature_array(doc) for doc in tqdm(val_all)])

# Linear regression model
def mse(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        sigma += np.square(pred[i] - gold[i])
    return sigma / (2 * pred.shape[0])

class LinearRegression:
    
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])
        
    def fit(self, X, Y):
        # Learn a model y = intercept + x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        # self.intercept = 0.0
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            # Thetas
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
        
    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            # s = self.intercept
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return Y
    
    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier for the final run.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y

# Test only
lr = LinearRegression(lrn_rate=0.1, n_iter=100)
lr.fit(x_train[:1000], np.array([(1 if i == 3 else 0) for i in y_train[:1000]]))

# Logistic regression functions
def sigmoid(Y):
    return 1 / (1 + np.exp(Y * -1))

def logistic_cost(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        if gold[i] == 1:  
            sigma -= np.log(pred[i])
        elif gold[i] == 0:
            sigma -= np.log(1 - pred[i])
    return sigma / pred.shape[0]

# Logistic regression model
class LogisticRegression:
    
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])
        
    def fit(self, X, Y):
        # Learn a model y = x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
        
    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return sigmoid(Y)
    
    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier for the final run.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y

# Test only
lr = LogisticRegression(lrn_rate=0.1, n_iter=100)
lr.fit(x_train[:1000], np.array([(1 if i == 3 else 0) for i in y_train[:1000]]))



# To use linear/logistic regression models to classify multiple classes
class OneVAllClassifier:
    
    def __init__(self, regression, **params):
        self.regression = regression
        self.params = params
        
    def fit(self, X, Y):
        # Learn a model for each parameter.
        self.categories = np.unique(Y)
        self.models = {}
        for cat in self.categories:
            ova_Y = np.array([(1 if i == cat else 0) for i in Y])
            model = self.regression(**self.params)
            model.fit(X, ova_Y)
            self.models[cat] = model
    
    def predict(self, X):
        # Predicts each x for each different model learned, and returns the category related to the model with the highest score.
        vals = {}
        for cat, model in self.models.items():
            vals[cat] = model.transform(X)
        Y = np.zeros(X.shape[0], dtype=np.int)
        for row in range(X.shape[0]):
            max_val, max_cat = -math.inf, -math.inf
            for cat, val in vals.items():
                if val[row] > max_val:
                    max_val, max_cat = val[row], cat
            Y[row] = max_cat
        return Y
    
# Test only
ova = OneVAllClassifier(LinearRegression, lrn_rate=0.1, n_iter=100)
ova.fit(x_train[:1000], y_train[:1000])

# Train the linear regression & One-V-All classifier models on the train set
clf = OneVAllClassifier(LinearRegression, lrn_rate=0.1, n_iter=1000)
clf.fit(x_train, y_train)

# Predict y for validation set
y_pred = clf.predict(x_val)
predicted = np.array([int_to_label[i] for i in y_pred])

# Prepare validation dataset format for score_submission in test.py
body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
gold_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])


test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))

# Predict y for validation set using logistic regression instead of linear regression, and compare results of scorer.py
clf_logistic = OneVAllClassifier(LogisticRegression, lrn_rate=0.1, n_iter=1000)
clf_logistic.fit(x_train, y_train)

y_pred = clf_logistic.predict(x_val)
predicted = np.array([int_to_label[i] for i in y_pred])

body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
gold_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])

test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
# linear regression performs better, so that model is chosen for the test set

# Load test data from CSV
f_tbodies = open('data/test_puls_body_texts.csv', 'r', encoding='utf-8')
csv_tbodies = csv.DictReader(f_tbodies)
tbodies = []
for row in csv_tbodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(tbodies):
        tbodies += [None] * (body_id + 1 - len(tbodies))
    tbodies[body_id] = row['articleBody']
f_tbodies.close()
tbody_inverse_index = {tbodies[i]: i for i in range(len(tbodies))}

test_all = []  

f_tstances = open('data/test_puls_stance.csv', 'r', encoding='utf-8')
csv_tstances = csv.DictReader(f_tstances)
for row in csv_tstances:
    body = tbodies[int(row['Body ID'])]
    test_all.append((row['target'], body, row['Stance']))
f_tstances.close()

# Initialise x (feature vectors) and y for test dataset
x_test = np.array([to_feature_array(doc) for doc in tqdm(test_all)])

# Predict y for test set
y_test = clf.predict(x_test)
pred_test = np.array([int_to_label[i] for i in y_test])

test_body_ids = [str(tbody_inverse_index[test_all[i][1]]) for i in range(len(test_all))]
test_pred_for_cm = np.array([{'target': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': pred_test[i]} for i in range(len(test_all))])
test_gold_for_cm = np.array([{'target': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': test_all[i][2]} for i in range(len(test_all))])

test_score, cm = score_submission(test_gold_for_cm, test_pred_for_cm)
null_score, max_score = score_defaults(test_gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))

score=[];names=[]

for model,name in classifiers:
    clf=model.fit(x_train,y_train)
    y_pred=clf.predict(x_val)
    predicted = np.array([int_to_label[i] for i in y_pred])
    body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
    pred_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
    gold_for_cm = np.array([{'target': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])
    
    test_score, cm = score_submission(gold_for_cm, pred_for_cm)
    null_score, max_score = score_defaults(gold_for_cm)
    print('*'*20);names.append(name)
    print(name)
    score.append(print_confusion_matrix(cm));
    a=SCORE_REPORT.format(max_score, null_score, test_score)

names=['LR','KNC','SVC','QDA','RFC','ADC','SGDC','DTC','XGB','LDA','GNB']
import seaborn as sns
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
A = score[:]


