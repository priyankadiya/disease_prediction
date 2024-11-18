import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.utils import shuffle

# Read and shuffle dataset
df = pd.read_csv('Health_dataset\dataset.csv')
df = shuffle(df, random_state=42)

# Replace underscores with spaces in column names
for col in df.columns:
    df[col] = df[col].str.replace('_', ' ')

# Print dataset statistics
print(df.describe())

# Check for null values before cleaning
null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print("Null values before cleaning:")
print(null_checker)

# Plot null values before cleaning
plt.figure(figsize=(10, 5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(rotation=45, ha='right')
plt.title('Before removing Null values')
plt.xlabel('Column Names')
plt.ylabel('Count')
plt.show()

# Strip whitespaces from data
cols = df.columns
data = df[cols].values.flatten()
s = pd.Series(data).str.strip().values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)

# Fill missing values with 0
df = df.fillna(0)

# Load symptom severity data
df1 = pd.read_csv('Health_dataset\Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_', ' ')

# Map symptom names to their weights
vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

df = pd.DataFrame(vals, columns=cols)

# Replace specific symptoms with 0
df = df.replace(['dischromic  patches', 'spotting  urination', 'foul smell of urine'], 0)

# Check for null values after cleaning
null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print("Null values after cleaning:")
print(null_checker)

# Plot null values after cleaning
plt.figure(figsize=(10, 5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(rotation=45, ha='right')
plt.title('After removing Null values')
plt.xlabel('Column Names')
plt.ylabel('Count')
plt.show()

# Print unique symptoms and diseases
print("Unique symptoms used:", len(df1['Symptom'].unique()))
print("Unique diseases identified:", len(df['Disease'].unique()))

# Prepare training and testing data
data = df.iloc[:, 1:].values
labels = df['Disease'].values
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=42)

# Train Random Forest model
rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators=500, max_depth=13)
rnd_forest.fit(x_train, y_train)

# Predict and evaluate model
preds = rnd_forest.predict(x_test)
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=np.unique(labels), columns=np.unique(labels))

# Print F1-score and accuracy
print('F1-score% =', f1_score(y_test, preds, average='macro') * 100)
print('Accuracy% =', accuracy_score(y_test, preds) * 100)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, preds))

# Cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
rnd_forest_train = cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
print("Cross-Validation Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_train.mean() * 100.0, rnd_forest_train.std() * 100.0))

rnd_forest_test = cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
print("Test Set Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_test.mean() * 100.0, rnd_forest_test.std() * 100.0))

# Save the model using joblib
joblib.dump(rnd_forest, './Health_Disease_RandomForestModel.joblib')

# Load the model
loaded_rf = joblib.load('./Health_Disease_RandomForestModel.joblib')

# Load symptom description and precautions data
discrp = pd.read_csv("Health_dataset\symptom_Description.csv")
prec = pd.read_csv("Health_dataset\symptom_precaution.csv")

# Function to predict disease based on symptoms
def predd(x, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
    psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    psy = [psymptoms]
    pred2 = x.predict(psy)

    # Print disease description and precautions
    disp = discrp[discrp['Disease'] == pred2[0]].values[0][1]
    recomnd = prec[prec['Disease'] == pred2[0]]
    c = np.where(prec['Disease'] == pred2[0])[0][0]
    precaution_list = []
    for i in range(1, len(prec.iloc[c])):
        precaution_list.append(prec.iloc[c, i])

    print("The Disease Name: ", pred2[0])
    print("The Disease Description: ", disp)
    print("Recommended Precautions: ")
    for i in precaution_list:
        print(i)

# Example prediction with symptoms from df1
sympList = df1["Symptom"].to_list()
predd(loaded_rf, sympList[56], sympList[66], sympList[15], sympList[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Save the model using pickle
filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rnd_forest, file)

