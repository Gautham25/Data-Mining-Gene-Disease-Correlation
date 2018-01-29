import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#all imports required for the following functions

headers = ["MAPPED_GENE","ANCESTRY","DISEASE/TRAIT"] # headers for the input data

df1 = pd.read_csv('Train.csv', sep=",", header=None,names=headers, na_values="?") #Reading the data
target_pred = pd.read_csv('prediction_answer.csv', sep="\t", header=None)  #Reading the correct value of validation of model

obj_df1 = df1.select_dtypes(include=["object"]).copy() #converting the input into objects
#changing from categorical values to numerical values
obj_df1["ANCESTRY"] = obj_df1["ANCESTRY"].astype('category')
obj_df1["ANCESTRY"] = obj_df1["ANCESTRY"].cat.codes
obj_df1["MAPPED_GENE"] = obj_df1["MAPPED_GENE"].astype('category')
obj_df1["MAPPED_GENE"] = obj_df1["MAPPED_GENE"].cat.codes
a = obj_df1[["MAPPED_GENE","ANCESTRY"]].copy() # copying features in a
b = obj_df1["DISEASE/TRAIT"].copy() # copying class in b

X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.3, shuffle=False)  #splitting in 0.7:0.3 - Train data:Test data

neigh = KNeighborsClassifier(n_neighbors=55, weights='distance', algorithm='auto', leaf_size=2, p=2, metric='minkowski', metric_params=None, n_jobs=1) #parameters for the decision tree model
neigh=neigh.fit(X_train, y_train)  #fitting the data to the model
score=neigh.score(X_test,y_test) #finding the score value with the test data
print("Accuracy : ", score*100,"%") #displaying the score value
c = pd.read_csv('prediction.csv', sep="\t", header=None) #reading the validation data
predicted = neigh.predict(c) #finding the prediction for the given validation data
r = recall_score(target_pred, predicted, average='macro') #finding the recall values
print("Recall : ", r) #displaying the recall of the model
print(*predicted,sep="\n") #displaying the predicted values