import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pandas import Series,DataFrame
from pylab import mpl
from sklearn import linear_model
data_train = pd.read_csv("Train.csv")
#print(data_train.columns)
#data_train.head()
#data_train.info()
#data_train.describe()

mpl.rcParams['font.sans-serif'] = ['FangSong'] 
mpl.rcParams['axes.unicode_minus'] = False 

fig = plt.figure()
fig.set(alpha=0.2)
data_train.Survived.value_counts().plot(kind='bar')
plt.title("savelife")
plt.ylabel("people")  
plt.show()
# print(data_train.Survived.value_counts())

data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("people")
plt.title("乘客等級分佈")
plt.show()
print(data_train.Pclass.value_counts())

plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")                         
plt.grid(True,axis='y') 
plt.title("savelife")
plt.show()

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("密度") 
plt.title("各等級的乘客年齡分佈")
plt.legend(('頭等艙', '2等艙','3等艙')) 
plt.show()

data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人數")
plt.ylabel("people")  
plt.show()

fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'saveY':Survived_1, 'saveN':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title("各乘客等級的獲救情況")
plt.xlabel("乘客等級") 
plt.ylabel("people") 
plt.legend()
plt.show()
# print(df)


fig = plt.figure()
fig.set(alpha=0.2) 
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("按性別看獲救情況")
plt.xlabel("性别") 
plt.ylabel("people")
plt.legend()
plt.show()
# print(df)

# Cousins
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df.head(8)

g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df.head()

#data_train.Cabin.value_counts().head(20)

fig = plt.figure()
fig.set(alpha=0.2) 
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'有':Survived_cabin, '無':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title("按Cabin有無看獲救情況")
plt.xlabel("Cabin") 
plt.ylabel("people")
plt.show()

data_train['Age']=data_train['Age'].fillna(data_train['Age'].mean())

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data_train = set_Cabin_type(data_train)

data_train['Embarked']=data_train['Embarked'].fillna('S')

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#df.head(10)

a=df.Age
df['Age_scaled'] = (a - a.mean()) / (a.std())
df=df.drop('Age',axis=1)
b=df.Fare
df['Fare_scaled'] = (b - b.mean()) / (b.std())
df=df.drop('Fare',axis=1)
#df.head(10)



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(penalty='l2', tol=1e-4)
clf.fit(X, y)  

print("模型正確率："+str(clf.score(X,y)))






