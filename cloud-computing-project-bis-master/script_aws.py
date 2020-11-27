import boto3
import json
import pandas as pd
import time

dataset = pd.read_csv("./datasets/original/dataset.csv")
dataset.reset_index(drop=True, inplace=True)
print( dataset['text'].describe() )

def authenticate_client():
    comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')
    return comprehend

comprehend = authenticate_client()

def sentiment_analysis(row, client=comprehend):
    if (row.name % 1000 == 0):
        if(row.name != 0):
            print(row.name)
            time.sleep(10)
    
    text = row.text
    
    #time_before = time.time()
    response = client.detect_sentiment(Text=text, LanguageCode='en')
    #print("time to get a response = " + str( (time.time() - time_before) ) )

    pred_sent = response["Sentiment"].lower()
    norm_mixed_score = response["SentimentScore"]["Mixed"]
    norm_pos_score = response["SentimentScore"]["Positive"]
    norm_obj_score = response["SentimentScore"]["Neutral"]
    norm_neg_score = response["SentimentScore"]["Negative"]
    norm_final_score = norm_pos_score - norm_neg_score
    
    return norm_obj_score, norm_mixed_score, norm_pos_score, norm_neg_score, norm_final_score, pred_sent   


df =  dataset
df[["norm_obj_score",
    "norm_mixed_score",
    "norm_pos_score",
    "norm_neg_score",
    "norm_final_score",
    "pred_sent"]] = dataset.apply(lambda x: sentiment_analysis(x), 
                                      axis='columns', 
                                      result_type='expand')

df.to_csv("./datasets/aws/dataset.csv")

print("done")
