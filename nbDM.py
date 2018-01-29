import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score

t_count = 0
adict = {}
adict_r = {}
adict_a = {}
dcount = {}
gcount = {}
acount = {}
ds_caused = []
dcy_count = {}
ay_count = {}
totalDisease = 0

headers = ["MAPPED_GENE", "ANCESTRY", "DISEASE/TRAIT"]

df1 = pd.read_csv("TRiM2.tsv", sep="\t", header=None,names=headers, na_values="?")


obj_df1 = df1.select_dtypes(include=["object"]).copy()

# Encode both features Gene and Ancestry to be provided as input to model
obj_df1["MAPPED_GENE"] = obj_df1["MAPPED_GENE"].astype('category')
obj_df1["MAPPED_GENE_cat"] = obj_df1["MAPPED_GENE"].cat.codes
obj_df1["ANCESTRY"] = obj_df1["ANCESTRY"].astype('category')
obj_df1["ANCESTRY_cat"] = obj_df1["ANCESTRY"].cat.codes

# Copy features to enable splitting
a = obj_df1[["MAPPED_GENE_cat","ANCESTRY_cat"]].copy()
b = obj_df1["DISEASE/TRAIT"].copy()
# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.3, shuffle=True)
arr = []
for i,j in zip(obj_df1["MAPPED_GENE_cat"], obj_df1["ANCESTRY_cat"]):
    new=[]
    new.append(i)
    new.append(j)
    arr.append(new)
arr2 = []

for i in obj_df1["DISEASE/TRAIT"]:
    arr2.append(i)

# Train Naive Bayes Model with input data
model = GaussianNB()
model.fit(arr, arr2)
target_test=[]
target_pred=[]

# Store actual values to test against predicted
for i,j in zip(obj_df1["MAPPED_GENE_cat"], obj_df1["ANCESTRY_cat"]):
    new=[]
    new.append(i)
    new.append(j)
    target_test.append(new)
for i in obj_df1["DISEASE/TRAIT"]:
    target_pred.append(i)
accuracy = 0
# Predict values based on test data
predicted = model.predict(target_test)
acc = 0
# Calculate Accuracy and recall
acc = model.score(X_test, y_test)
print("\nACCURACY =", acc*100, "%")
r = recall_score(target_pred, predicted, average='macro')
print("RECALL=", r*100, "%")