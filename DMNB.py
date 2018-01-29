import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
#all imports required for the following functions

headers = ["MAPPED_GENE","ANCESTRY","DISEASE/TRAIT"] # headers for the input data

#df1 = pd.read_csv("test.tsv", sep="\t", header=None,names=headers, na_values="?") #Reading the data
#df1 = pd.read_csv('TRiM2.tsv', sep="\t", header=None,names=headers, na_values="?") #Reading the data
df1 = pd.read_csv('Train.csv', sep=",", header=None,names=headers, na_values="?") #Reading the data

obj_df1 = df1.select_dtypes(include=["object"]).copy() #converting the input into objects

# Encode both features Gene and Ancestry to be provided as input to model
obj_df1["MAPPED_GENE"] = obj_df1["MAPPED_GENE"].astype('category')
obj_df1["MAPPED_GENE_cat"] = obj_df1["MAPPED_GENE"].cat.codes
obj_df1["ANCESTRY"] = obj_df1["ANCESTRY"].astype('category')
obj_df1["ANCESTRY_cat"] = obj_df1["ANCESTRY"].cat.codes
a = obj_df1[["MAPPED_GENE_cat","ANCESTRY_cat"]].copy() #copying features in a
b = obj_df1["DISEASE/TRAIT"].copy() #copying class in b

X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.3, shuffle=False) #splitting in 0.7:0.3 - Train data:Test data

arr = [] #grouping MAPPED GENE and ANCESTRY to arr.
for i,j in zip(obj_df1["MAPPED_GENE_cat"], obj_df1["ANCESTRY_cat"]):
    new=[]
    new.append(i)
    new.append(j)
    arr.append(new)

arr2 = [] #copying DISEASE/TRAIT to arr2.
for i in obj_df1["DISEASE/TRAIT"]:
    arr2.append(i)

model = GaussianNB() #parameters for the naive bayes model with default value
model.fit(arr, arr2)
#model.partial_fit(arr, arr2, np.unique(arr2)) #fitting the data to the model
target_test=[] #initialising
target_pred=[] #
for i,j in zip(obj_df1["MAPPED_GENE_cat"], obj_df1["ANCESTRY_cat"]):
    new=[]
    new.append(i)
    new.append(j)
    target_test.append(new) #data set for validation
for i in obj_df1["DISEASE/TRAIT"]:
    target_pred.append(i) #
accuracy = 0 #initialising
target_pre = pd.read_csv('prediction_answer.csv', sep="\t", header=None) #Reading the correct value of validation of model
c = pd.read_csv('prediction.csv', sep="\t", header=None) #reading the validation data
predicted = model.predict(c)  #finding the prediction for the given validation data
acc = 0 #initialising
acc = model.score(X_test, y_test) #finding the score value with the test data
print("\nACCURACY =", acc*100, "%") #displaying the score value
r = recall_score(target_pre, predicted, average='macro') #finding the recall values
print("RECALL=", r)  #displaying the recall of the model
print(*predicted,sep="\n") #displaying the predicted values