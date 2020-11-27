## Artifact check-list

Details: [Link](http://cTuning.org/ae/submission_extra.html)

* **Algorithm:** Tweet sentiment analysis
* **Program:** Scikit-Learn, AWS Comprehend, Azure Cognitive service, NLTK
* **Model:** SentiWordNet, Logistic Regression
* **Data set:** Included (train (~5500 tweets) tweets and test (~4000 tweets) set)
* **Run-time environment:** Mac OS
* **Hardware:** Intel core i7 CPU
* **Metrics:** Accuracy, F1 score, Confusion Matrix 
* **Output:** Classification Report
* **Experiments:** Compare classification result from three models
* **How much disk space required (approximately)?** ~4GB
* **Publicly available?:** Yes
* **How much time is needed to prepare workflow (approximately)?:** A few minutes
* **How much time is needed to complete experiments (approximately)?:** A few minutes
* **Code license(s)?:** MIT license 

## Installation

### Install global prerequisites (MacOS)

```
pip install boto3, joblib, json5, jsonschema, notebook, pandas, scikit-learn, re \
            numpy, matplotlib, pandas
```

### Install package for Microsoft Azure Cloud
```
pip install azure-ai-textanalytics, azure-common, azure-core
```

### Install package for AWS Cloud
```
pip install azure-ai-textanalytics, azure-common, azure-core
```
Follow this link to set-up your environemnt :
```
https://docs.aws.amazon.com/comprehend/latest/dg/get-started-api.html
```

## Experiment

### Run the experiment
```
bash run_experiment.sh
```


