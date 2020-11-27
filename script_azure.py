import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import time

dataset = pd.read_csv("./datasets/original/dataset.csv")
dataset.reset_index(drop=True, inplace=True)

key = "add-your-own-key"
endpoint = "https://sentiment-analysis-project.cognitiveservices.azure.com/"

def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

azure_client = authenticate_client()

def sentiment_analysis(row, client=azure_client):
    if (row.name % 1000 == 0):
        if(row.name != 0):
            print(row.name)
            # this is to make sure that Azure service will not receive too much request per minute
            time.sleep(30)
    
    documents = [row.text]
    
    #time_before = time.time()
    response = client.analyze_sentiment(documents = documents)[0]
    #print("time to get a response = " + str( (time.time() - time_before) ))
    
    pred_sent = response.sentiment
    norm_pos_score = response.confidence_scores.positive
    norm_obj_score = response.confidence_scores.neutral
    norm_neg_score = response.confidence_scores.negative
    norm_final_score = norm_pos_score - norm_neg_score
    
    return norm_obj_score, norm_pos_score, norm_neg_score, norm_final_score, pred_sent   



df =  dataset
df[["norm_obj_score",
   "norm_pos_score",
   "norm_neg_score",
   "norm_final_score",
   "pred_sent"]] = dataset.apply(lambda x: sentiment_analysis(x), 
                                      axis='columns', 
                                      result_type='expand')

df.to_csv("./datasets/microsoft/dataset.csv")

print(dataset.head())
