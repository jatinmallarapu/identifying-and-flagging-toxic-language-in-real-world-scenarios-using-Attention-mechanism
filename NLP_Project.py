#implementing model to detect toxic language in real-world scenarios.
#Since the daatset is large we can download using the kaggle functions

! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
!kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
! mkdir textclassification
! unzip jigsaw-toxic-comment-classification-challenge.zip -d textclassification
! unzip textclassification/train.csv.zip  import pandas as pd
df = pd.read_csv("textclassification/train.csv.zip")
## Taking sample to work on just for framework demonstration ##
df_data = df.sample(frac=0.2, replace=True, random_state=1)
df_data.isnull().sum()
import nltk
nltk.download('stopwords')
## Word Pre-Processing ##
import nltk
import string
import re
wpt = nltk.WordPunctTokenizer()
stop_words_init = nltk.corpus.stopwords.words('english')
stop_words = [i for i in stop_words_init if i not in ('not','and','for')]
print(stop_words)
## Function to normalize text for pre-processing ##
def normalize_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text
## Apply the written function ##
df_data['comment_text'] = df_data['comment_text'].apply(lambda x: normalize_text(x))
processed_list = []
for j in df_data['comment_text']:
    process = j.replace('...','')
    processed_list.append(process)
    
df_processed = pd.DataFrame(processed_list)
df_processed.columns = ['comments']
df_processed.head(n=5)
#Adding labels and ploting the graph
import seaborn as sns
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = df_data[labels].values
import matplotlib.pyplot as plt
val_counts = df_data[labels].sum()
plt.figure(figsize=(8,5))
ax = sns.barplot(val_counts.index, val_counts.values, alpha=0.8)
plt.title("Labels per Classes")
plt.xlabel("Various Label Type")
plt.ylabel("Counts of the Labels")
rects = ax.patches
labels = val_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha="center", va="bottom")
plt.show()
#The dataset is imbalanced as If you check the ratio between DEATH_EVENT=1 and DEATH_EVENT=0, it is 2:1 which means our dataset is imbalanced.
import seaborn as sns
fig , axes = plt.subplots(2,3,figsize = (10,10), constrained_layout = True)
sns.countplot(ax=axes[0,0],x='toxic',data=df_data )
sns.countplot(ax=axes[0,1],x='severe_toxic',data=df_data)
sns.countplot(ax=axes[0,2],x='obscene',data=df_data)
sns.countplot(ax = axes[1,0],x='threat',data=df_data)
sns.countplot(ax=axes[1,1],x='insult',data=df_data)
sns.countplot(ax=axes[1,2],x='identity_hate',data=df_data)
plt.suptitle('Number Of Labels of each Toxicity Type')
plt.show()

! kaggle datasets download danielwillgeorge/glove6b100dtxt
! mkdir glove
! unzip glove6b100dtxt.zip -d glove
X = list(df_processed['comments'])
y_data = df_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
y = y_data.values
#Initially I have split data set into training and testing values
#Later I have used the training and testing values in spliting the validation data values

#Tokenizing the data and applying pad sequencing
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional,GRU,concatenate,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,Conv1D
from keras.models import Model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers import concatenate
import matplotlib.pyplot as plt
from keras import layers
from keras.optimizers import Adam,SGD,RMSprop
######## Textual Features for Embedding ###################
max_len_m1 = 100
max_features_m1 = 10000
embed_size_m1 = 300
tokenizer = Tokenizer(num_words=max_features_m1)
tokenizer.fit_on_texts(list(x_train)+list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test= tokenizer.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, padding='post', maxlen=max_len_m1)
x_test = pad_sequences(x_test, padding='post', maxlen=max_len_m1)
embeddings_dictionary = dict()
glove_file = open("/content/glove/glove.6B.100d.txt", encoding="utf8") ## pre-trained or self trained global vectors file ##
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()
from numpy import array
from numpy import asarray
from numpy import zeros
embeddings_dictionary_m1 = dict()

vocab_size = len(tokenizer.word_index) + 1  ## total distinct words is the Vocabulary ##
word_index = tokenizer.word_index
num_words = min(max_features_m1,len(word_index)+1)
embedding_matrix = zeros((num_words, embed_size_m1)) ## has to be similar to glove dimension ##
for word, index in tokenizer.word_index.items():
    if index >= max_features_m1:
        continue
    embedding_vector = embeddings_dictionary_m1.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
#Creating a CNN model and Bilstm model for Multi text classification
#Here I have used RMS propagation optimizer and cross entropy loss function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
sequence_input = Input(shape=(max_len_m1, ))
x=Attention(return_sequences=True)
x = Embedding(max_features_m1, embed_size_m1, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x) ## ostly drops the entire 1D feature map rather than individual elements.
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
avg_pool = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(avg_pool)
x = Dropout(0.2)(x)

preds = Dense(6, activation="sigmoid")(x)

m1 = Model(sequence_input, preds)
#model.add(Attention(return_sequences=False))
m1.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=1e-3),metrics=['accuracy'])
print(m1.summary())
#Obtain accuracy is 0.9941 for attention mechanism CNN-Bilstm model by using glove file
h1 = m1.fit(x_train, y_train, batch_size=128, epochs=5,verbose=1, validation_split=0.2)

m1.save_weights("./BiLSTM_ver1.h5")
print(f"Accuracy: {score_m1[1]}")
y_pred = m1.predict(x_test)
len(y_test)
#y_pred=model.predict(X_test) 
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print("Precision Score : ",precision_score(y_test, y_pred, pos_label='positive',average='micro'))
print('Recall: %.3f' % recall_score(y_test, y_pred,pos_label='positive',average='micro'))
print('F1 Score: %.5f' % f1_score(y_test, y_pred,pos_label='positive',average='micro'))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_scorefrom sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.15, train_size=0.85)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import Bidirectional,GRU,concatenate,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,Conv1D
from keras.models import Model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers import concatenate
import matplotlib.pyplot as plt
from keras import layers
from keras.optimizers import Adam,SGD,RMSprop
######## Textual Features for Embedding ###################
max_len_m1 = 100
max_features_m1 = 10000
embed_size_m1 = 300
tokenizer = Tokenizer(num_words=max_features_m1)
tokenizer.fit_on_texts(list(x_train)+list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test= tokenizer.texts_to_sequences(x_test)
x_train = pad_sequences(x_train, padding='post', maxlen=max_len_m1)
x_test = pad_sequences(x_test, padding='post', maxlen=max_len_m1)

import sklearn.metrics as metrics
pred_prob = m1.predict_on_batch(x_test)
fpr_list = []
tpr_list = []
threshold_list = []
roc_auc_list = []
required_columns = (df_data.shape[1])-2
for i in range(6): # you can make this more general
    fpr, tpr, threshold = metrics.roc_curve(y_test[:, 0], pred_prob[:, 0])
    roc_auc = metrics.auc(fpr, tpr)
    
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)
    roc_auc_list.append(roc_auc)
#This ROC curve has an AUC = 0.5, which meaning it ranks a random positive example higher than a random negative example less than 50% of the time.
print(f"The AUC score for the ROC curve is : {roc_auc}")
import matplotlib.pyplot as plt
for val in range(1):
  plt.plot(fpr_list[val], tpr_list[val],color='red')
  #plt.plot( threshold_list[val],linestyle='--',color='orange', label='Class 0 vs Rest')
plt.title(f"ROC-AUC Curve with AUC score: {roc_auc}")
plt.xlabel("False positive rate")
plt.ylabel("True positiv rate")
plt.show()

loss_train = h1.history['loss']
loss_val = h1.history['val_loss']
plt.plot(h1.epoch, loss_train, 'g', label='Training loss')
plt.plot(h1.epoch, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = h1.history['accuracy']
loss_val = h1.history['val_accuracy']
epochs = range(1,11)
plt.plot(h1.epoch, loss_train, 'g', label='Training accuracy')
plt.plot(h1.epoch, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
