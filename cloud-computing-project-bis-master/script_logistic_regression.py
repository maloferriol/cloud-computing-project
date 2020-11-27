import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
import string  
import re
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import model_from_json

from joblib import dump, load

def preprocessing(sentences):
    lemmatizer = WordNetLemmatizer() 
    
    words = []
     # regex expression to extract URL
    for sent in sentences:
        result = re.sub(r"http\S+", "", sent)
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", result)
        
        token = word_tokenize(clean)
        tokens = ''
        for t in token:
            if (    (t not in stopwords.words('english')) # remove stop words
                and (t not in string.punctuation) # remove punctuation
                and (re.match(r"\b[a-zA-Z]{3,}\b",t)) # keep only the words with more than three character 
                and (re.match(r"(?u)\b\w\w+\b",t)) # regex expression from scikit tokeniser
                    ): 
                tokens += ' ' + lemmatizer.lemmatize(t.lower())
        words.append(tokens)
      
    return words

def catnames_to_number(sentiment):
    new_dataset = ""
    if (sentiment == "positive"):
        return 0
    elif (sentiment == "neutral"):
        return 1
    elif (sentiment == "negative"):
        return 2
    else :
        print("error in catnames_to_number")
    return new_dataset


cntvect = feature_extraction.text.CountVectorizer( 
    input='content',
    encoding="utf-8",
    decode_error="strict", 
    strip_accents="unicode",
    lowercase=True, 
    analyzer='word',
    stop_words="english"
)

tf_transformer = feature_extraction.text.TfidfTransformer()

parameters = {
    'vect__max_features': (1000,2500, 5000),
    'vect__ngram_range': ((1, 1), (1, 2)),
    
    'tfidf__norm': ('l1','l2'),
    'tfidf__use_idf': (False,True),

    'clf__alpha': geomspace(1e-2, 1e1, num=10)
}

classNames = ['positive', 'neutral','negative'] 

pipe = pipeline.Pipeline([
    ('vect', cntvect),
    ('tfidf', tf_transformer),
    ('clf', linear_model.LogisticRegression())
])

gs = model_selection.GridSearchCV(pipe, parameters, cv = 5, n_jobs=-1,verbose=0)

new_df = pd.read_csv("./datasets/project_dataset/dataset.csv", sep=';')
new_df = new_df.set_index('textID')

train_gt_X_lem = preprocessing(train_gt_X)

gs.fit(train_gt_X_lem, train_gt_Y); 

lp_pipe_model_gt = gs.best_estimator_

y_gt_test_pred = lp_pipe_model_gt.predict(test_X)

new_report = classification_report(test_Y, y_gt_test_pred,target_names=classNames, output_dict=True)

# Read data from file:
previous_report = json.load( open( "models/report.json" ) )

if (new_report["accuracy"] > previous_report["accuracy"]):
    filename_lr_model_gt = 'models/lr_model.joblib'
    dump(lp_pipe_model_gt, filename_lr_model_gt) 
    # Serialize data into file:
    json.dump( new_report, open( "models/report.json", 'w' ) )



