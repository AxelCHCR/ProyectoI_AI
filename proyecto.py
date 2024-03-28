import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
# Model Performance Analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('Datasets/diabetes.csv')
print(dataset.head())

dataset.info()
dataset.describe()

data_copy = dataset.copy(deep = True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(data_copy.isnull().sum())

data_copy['Glucose'] = data_copy['Glucose'].fillna(data_copy['Glucose'].mean())
data_copy['BloodPressure'] = data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean())
data_copy['SkinThickness'] = data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median())
data_copy['Insulin'] = data_copy['Insulin'].fillna(data_copy['Insulin'].median())
data_copy['BMI'] = data_copy['BMI'].fillna(data_copy['BMI'].median())

p=data_copy.Outcome.value_counts().plot(kind="bar")

p = data_copy.hist(figsize = (20,20))

sns.displot(data_copy['BloodPressure'], kind = 'kde')
plt.show()

# El valor de "BMI" para la persona con el valor de "Glucose" más alto
data_copy[data_copy['Glucose'] == data_copy['Glucose'].max()]['BMI']

# ¿Cuántas mujeres tienen la glucosa por encima de la media?
data_copy[data_copy['Glucose'] > data_copy['Glucose'].mean()].shape[0]

# ¿Cuántas personas tienen un nivel de glucosa mayor al promedio y un índice de masa corporal menor al promedio?
dataset[(dataset['BloodPressure'] == dataset['BloodPressure'].median()) & (dataset['BMI'] < dataset['BMI'].median())].shape[0]

sns.pairplot(data = data_copy, vars = ['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue = 'Outcome')
plt.show()

sns.scatterplot(x = 'Glucose', y = 'Insulin', data = data_copy)
plt.show()

plt.boxplot(data_copy['Age'])
plt.title('Boxplot de la variable Age')
plt.ylabel('Age')
plt.show()

plt.hist(data_copy[data_copy['Outcome'] == 1]['Age'], bins = 5)
plt.title('Distribution of Age for Women who has Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.hist(data_copy[data_copy['Outcome'] == 0]['Age'], bins = 5)
plt.title('Distribution of Age for Women who do not have Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sns.heatmap(data_copy.corr(), annot=True)
plt.show()

X = data_copy.drop('Outcome', axis=1) # Features
y = data_copy['Outcome'] # Target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training and Prediction using KNN
test_scores = []
train_scores = []

for i in range(1,100):
    knn = KNN(n_neighbors=i)
    knn.fit(x_train,y_train)
    
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))

max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

plt.figure(figsize=(12, 5))
p = sns.lineplot(x=range(1, 100), y=train_scores, marker='*', label='Train Score')
p = sns.lineplot(x=range(1, 100), y=test_scores, marker='o', label='Test Score')

knn = KNN(28)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)

# Model Performance Analysis
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="RdYlGn" ,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test,y_pred))

y_pred_proba = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=28) ROC curve')
plt.show()

roc_auc_score(y_test,y_pred_proba)

param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNN()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)
print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))