import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv('heartDisease_2020_sampling.csv')

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

# Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "GenHealth", "AlcoholDrinking", "AgeCategory", "Smoking", "PhysicalActivity"])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Sex", "Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())

input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split

X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# print(X_train.head())
# print(X_test.head())
# print(y_train.head())
# print(y_test.head())

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 10, class_weight = "balanced")
clf = clf.fit(X_train, y_train)
test_preds = clf.predict(X_test)

#Test the model with the testing data set and prints accuracy score
from sklearn.metrics import accuracy_score

test_acc = accuracy_score(y_test, test_preds)

print("The accuracy score of the Decision Tree Model from testing is: " + str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_preds, labels = [1, 0])
print("The confusion matrix of the tree is:")
print(cm)


#Test the model with the training data set and prints accuracy score
train_preds = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)

print("The accuracy score of the Decision Tree Model from training is: " + str(train_acc))



input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations

print("Decisions trees could also be used to predict whether a credit card transaction is fraudulent or not")
print("To ensure the model performs fairly, the data should be balanced to include both fraudulent and non-fraudulent transactions." )
print("The data set also should be adequately split into training and testing data sets to ensure the model can learn during training while also allowing for plenty of test points to more accurately evaluate the model.")

#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)
