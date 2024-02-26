#Import necessary packages

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score,classification_report

data = pd.read_csv("RIVM_labelled_corpus.csv")
data.head(2)

#Preprocess all the text
def preprocess(text):
    text = text.lower()
    text = re.sub("\n"," ",text) #Remove all next lines
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëėęûüùúūôöòóõœøîïíīįìàáâäæãåçćč&@#A-ZÇĆČÉÈÊËĒĘÛÜÙÚŪÔÖÒÓŒØŌÕÎÏÍĪĮÌ0-9- \']', "", text) #Remove special characters
    text = re.sub("\?","question_mark",text) #Replace question mark as mentioned in the paper
    #wrds = text.split()
    return text

# Drop any rows from the DataFrame data that contain missing values (NaN) and split the labels into a list of labels
data = data.dropna()
data["new labels"] = data["labels"].apply(lambda row:str(row).split(","))

# Remove rows where labels occur only once
value_counts = data["new labels"].value_counts()
data = data[data['new labels'].isin(value_counts[value_counts != 1].index)]

# Apply the preprocessing function to the data column
data['content of post preprocessed'] = data['content of post'].apply(preprocess)

# Convert data into numerical format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(list(data["new labels"]))

# Vectorization of data contents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['content of post preprocessed'])

# Splitting the dataset while maintaining the label balance
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

# Print shapes to verify the split
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

clf = OneVsRestClassifier(LinearSVC(C=1.0, max_iter=10000, random_state=42))

# Train the model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
f1_score = f1_score(y_test, y_pred, average =None)
print("f1_score:", f1_score)

# Compute precision, recall, and F1 score for each label
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

# Get the label names
label_names = mlb.classes_

# Print results for each label
for i, label in enumerate(label_names):
    print(f"Label: {label}")
    print(f"  Precision: {precision[i]}")
    print(f"  Recall: {recall[i]}")
    print(f"  F1 Score: {f1_score[i]}")
    print()

print(classification_report(y_test, y_pred,target_names=label_names))