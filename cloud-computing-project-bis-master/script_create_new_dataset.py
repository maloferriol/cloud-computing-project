import pandas as pd 
import time

aws_columns = {
    "norm_obj_score":  "aws_obj_score",
    "norm_mixed_score":"aws_mixed_score",
    "norm_pos_score":  "aws_mixed_score",
    "norm_neg_score":  "aws_neg_score",
    "norm_final_score":"aws_final_score",
    "pred_sent":  "aws_pred_sent"
}

azure_columns = {
    "norm_obj_score":  "azure_obj_score",
    "norm_pos_score":  "azure_pos_score",
    "norm_neg_score":  "azure_neg_score",
    "norm_final_score":"azure_final_score",
    "pred_sent":  "azure_pred_sent"
}

comparision_cols = [
    "sent_vs_azure",
    "sent_vs_aws",
    "aws_vs_azure",
    "aws_vs_azure_mixed", 
    "all_true_sent",
    "all_false_sent",
    "same_but_false",
    "azure_t_vs_aws_f",
    "aws_t_vs_azure_f"]

def catnames_to_number_in_row(row):
    # We label the sentiment with -1 when we don't want to keep that sample
    new_sent_value = -1
    if ((row.azure_pred_sent == "positive") and (row.aws_pred_sent == "positive")):
        new_sent_value = 0
    elif ((row.azure_pred_sent == "neutral") or (row.azure_pred_sent == "mixed") and 
         (row.aws_pred_sent == "neutral") or (row.aws_pred_sent == "mixed")):
        new_sent_value = 1
    elif ((row.azure_pred_sent == "negative") and (row.aws_pred_sent == "negative")):
        new_sent_value = 2
    return new_sent_value

def sentiment_analysis(row):
    sent_vs_azure = row.azure_pred_sent == row.sentiment
    sent_vs_aws   = row.sentiment == row.aws_pred_sent
    aws_vs_azure  = row.azure_pred_sent == row.aws_pred_sent

    aws_vs_azure_mixed = (row.azure_pred_sent == row.aws_pred_sent) and (row.azure_pred_sent == "mixed")

    all_true_sent  = ((row.azure_pred_sent == row.aws_pred_sent) and (row.azure_pred_sent == row.sentiment))
    all_false_sent = ((row.azure_pred_sent != row.sentiment)     and (row.aws_pred_sent != row.sentiment))

    same_but_false   = ((row.azure_pred_sent != row.sentiment)     and (row.aws_pred_sent == row.azure_pred_sent))
    azure_t_vs_aws_f = ((row.azure_pred_sent != row.aws_pred_sent) and (row.azure_pred_sent == row.sentiment))
    aws_t_vs_azure_f = ((row.azure_pred_sent != row.aws_pred_sent) and (row.aws_pred_sent == row.sentiment))

    return sent_vs_azure, sent_vs_aws, aws_vs_azure, aws_vs_azure_mixed, all_true_sent, all_false_sent, same_but_false, azure_t_vs_aws_f, aws_t_vs_azure_f


###################################### AWS ##############################################

dataset_aws = pd.read_csv("./datasets/aws/dataset.csv")
dataset_aws.reset_index(drop=True, inplace=True)
dataset_aws = dataset_aws.drop(columns=[ "Unnamed: 0", "Unnamed: 0.1"])
dataset_aws.rename(aws_columns, axis="columns", inplace=True)

##################################### Azure #############################################

dataset_azure = pd.read_csv("./datasets/microsoft/dataset.csv")
dataset_azure.reset_index(drop=True, inplace=True)
dataset_azure = dataset_azure.drop(columns=["text","sentiment", "Unnamed: 0", "Unnamed: 0.1"])
dataset_azure.rename(azure_columns, axis="columns", inplace=True)


dataset = dataset_aws.merge(dataset_azure, how='inner', on='textID', validate='one_to_one')

############################# Create dataset for analysis ###############################

dataset_analysis = dataset
dataset_analysis[comparision_cols] = dataset.apply(lambda x: sentiment_analysis(x), 
                                      axis='columns', 
                                      result_type='expand')

dataset_analysis.to_csv("./datasets/analysis/dataset.csv", sep=";")

############################# Create dataset with new label ##############################

dataset["new_sentiment"] = dataset.apply(catnames_to_number_in_row, axis=1)

cloud_based_dataset = dataset[['textID','text','new_sentiment']]
cloud_based_dataset = cloud_based_dataset.rename({'new_sentiment':'sentiment'}, axis=1)
cloud_based_dataset = cloud_based_dataset[cloud_based_dataset["sentiment"] != -1]

cloud_based_dataset.to_csv("./datasets/project_dataset/dataset.csv", sep=";")
