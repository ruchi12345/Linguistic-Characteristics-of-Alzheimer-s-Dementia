from scipy import sparse
from keras.optimizers import Adam, Adamax, Nadam, Adadelta
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.utils import resample
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # -1 !!!!
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import nltk, re, pprint
nltk.download('all')
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
import string
col_names = ['label','message']
msg_frame1 = pd.DataFrame()
msg_frame2 = pd.read_csv("filter_output_file.csv")
msg_frame2.head()
msg=msg_frame2

def process_msg(msg):
        msg = msg.lower()
        msg = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',msg)
        msg = re.sub('[\s]+', ' ', msg)
        msg = msg.translate(str.maketrans('','', string.punctuation))
        msg = msg.strip()
        msg = msg.rstrip('\'"')
        msg = msg.lstrip('\'"')
        return msg

msg['message'][0].translate(str.maketrans('','', string.punctuation))

def get_uppercase_ratio(msg):
    count_of_upperchar = sum(1 for c in msg if c.isupper())
    length_of_sms =  len(msg) - msg.count(' ')
    return  count_of_upperchar/length_of_sms

msg["uppercase_ratio"] = msg["message"].map(lambda text: get_uppercase_ratio(text))

def get_nonalphanumeric_ratio(msg):
    count_of_nonalnum = sum(1 for c in msg if not c.isalnum())
    length_of_sms =  len(msg) - msg.count(' ')
    return count_of_nonalnum/length_of_sms

msg["nonalnum_ratio"] = msg["message"].map(lambda text: get_nonalphanumeric_ratio(text))

def get_numeric_ratio(msg):
    count_of_numeric_chars = sum(1 for c in msg if c.isdigit())
    length_of_sms =  len(msg) - msg.count(' ')
    return count_of_numeric_chars/length_of_sms

msg["numeric_ratio"] = msg["message"].map(lambda text: get_numeric_ratio(text))

def has_url(msg):
    if 'http' in msg or 'www' in msg:
        return 1
    else:
        return 0

msg["has_url"] = msg["message"].map(lambda text: has_url(text))

processed_msg = [ ]

for each_sms in msg[ 'message' ]:
    psms = process_msg (each_sms)
    processed_msg.append (psms)

msg["message"] = processed_msg

def split_to_tokens(message):
    return word_tokenize(message)
msg["no_of_terms"] = msg["message"].map(lambda text: len(split_to_tokens(text)))
msg["length"] = msg["message"].map(lambda text: len(text))

def get_pos_ratio(msg):
    word_tokens = word_tokenize(msg)
    word_tokens_len = len(word_tokens)
    if word_tokens_len == 0:
        return (0,0,0,0,0)
    tagged_words = nltk.pos_tag(word_tokens)
    NOUN_count = len(list(filter(lambda x:x[1] in ['NN', 'NNP', 'NNS', 'NNPS'], tagged_words)))
    VERB_count = len(list(filter(lambda x:x[1] in ['VB', 'VBZ', 'VBD', 'VBN', 'VBP', 'VBG'], tagged_words)))
    PRONOUN_count = len(list(filter(lambda x:x[1] in ['PRP', 'PRP$'], tagged_words)))
    MODIFIER_count = len(list(filter(lambda x:x[1] in ['JJ', 'JJR', 'JJS'], tagged_words)))
    WH_count = len(list(filter(lambda x:x[1] in ['WP', 'WP$', 'WDT', 'WRB'], tagged_words)))

    return (NOUN_count/word_tokens_len,
            VERB_count/word_tokens_len,
            PRONOUN_count/word_tokens_len,
            MODIFIER_count/word_tokens_len,
            WH_count/word_tokens_len)

msg["NOUN_ratio"], msg["VERB_ratio"], msg["PRONOUN_ratio"], msg["MODIFIER_ratio"], msg["WH_ratio"]\
= zip(*msg["message"].map(get_pos_ratio))

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

df_majority = msg[msg.Label == 'control']
df_minority = msg[msg.Label == 'dementia']
print(len(df_majority))
print(len(df_minority))
df_minority_upsampled = resample ( df_minority , replace=True ,  # sample with replacement
                                   n_samples=2181 ,  # to match majority class
                                   random_state=123 )  # reproducible results
df_upsampled = pd.concat ( [df_majority , df_minority_upsampled] )
print(df_upsampled.Label.value_counts ())

le = preprocessing.LabelEncoder()


X = df_upsampled.drop('Label', axis=1)
y = df_upsampled.Label
le.fit(y)
Y=le.transform(y)

vect = CountVectorizer(analyzer=split_into_lemmas, stop_words='english')
X_train_message_vect = vect.fit_transform(X.message)
X_train_message_tfidf = TfidfTransformer().fit_transform(X_train_message_vect)
X_train_numeric = X.iloc[:, 2:]
combined_features_train = sparse.hstack((X_train_message_tfidf, X_train_numeric)).toarray()
X_train, X_test, y_train, y_test = np.array(train_test_split(combined_features_train, Y, test_size=0.20, random_state=42))
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
input_shape = X_train.shape
input_tensor = Input((1310,1,))
x = input_tensor
rnn_size=32
x =  ZeroPadding1D((1))(input_tensor)
x = Convolution1D (32, 3) (x)
x = Dropout (0.25) (x)
x = Activation ('relu') (x)
x = MaxPooling1D(pool_size=(2)) (x)
x =  ZeroPadding1D((1))(x)
x = Convolution1D (32, 3) (x)
x = Dropout (0.5) (x)
x = Activation ('relu') (x)
x = MaxPooling1D(pool_size=(2)) (x)

x =  ZeroPadding1D((1))(x)
x = Convolution1D (64, 3) (x)
x = Activation ('relu') (x)
x = Dropout (0.25) (x)
x = MaxPooling1D(pool_size=(2)) (x)
x =  ZeroPadding1D((1))(x)
x = Convolution1D (64, 3) (x)
x = Activation ('relu') (x)
x = Dropout (0.25) (x)
x = MaxPooling1D(pool_size=(2)) (x)
x =  ZeroPadding1D((1))(x)
x = Convolution1D (128, 3) (x)
x = Activation ('relu') (x)
x = Dropout (0.25) (x)
x = MaxPooling1D(pool_size=(2)) (x)
x = ZeroPadding1D((1))(x)
x = Convolution1D (128, 3) (x)
x = Activation ('relu') (x)
x = Dropout (0.25) (x)
x = MaxPooling1D(pool_size=(2)) (x)
from keras.layers import merge
gru_1 = LSTM (rnn_size, return_sequences=True, init='he_normal', name='gru1') (x)
gru_2 = LSTM (rnn_size, return_sequences=True, init='he_normal', name='gru2') (gru_1)
x = Flatten ()(gru_2)
x = Dense(1026)(x)
x = Dropout (0.5) (x)
x = Dense(1, activation='sigmoid',kernel_initializer='normal')(x)

model = Model(input=input_tensor, output=x)
Adam = Adam(lr=0.000034)
model.compile(loss='binary_crossentropy',
          optimizer=Adam,metrics=['accuracy'])
from IPython.display import Image
Image('mBdel.png')
plt.show()
ll=model.fit(
    X_train, y_train,
    batch_size=90, nb_epoch=2,verbose=2,validation_data=(X_test,y_test)
)

main_val=ll.history
total_epoch=ll.epoch




