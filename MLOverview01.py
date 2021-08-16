import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import os

allData= pd.read_csv((os.path.join(os.getcwd(),'dataSet_Eligibility.csv')) )
#C:\Users\Toto A\___pyLEARNING\kaggle\LinkedIn\allData.csv
features=['Id_No', 'Sex', 'Age',  'Height', 'Weight'] 
X = allData[features]
y = allData['Eligible']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0, shuffle=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()