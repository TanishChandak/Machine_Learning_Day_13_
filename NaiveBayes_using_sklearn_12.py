import pandas as pd

df = pd.read_csv('titanic.csv')
# print(df.head())

# Removing un-necessary columns:
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis='columns', inplace=True)
print(df.head())

# Dependent variables:
target = df.Survived
print(target.head())

# Independent variables:
inputs = df.drop('Survived', axis='columns')
print(inputs.head())

# creting the dummy variables of the (Sex) column:
dummies = pd.get_dummies(inputs.Sex)
print(dummies.head())

# merging the dummies variables into the inputs data:
inputs = pd.concat([inputs, dummies], axis='columns')
print(inputs.head())

# Removing the SEX column:
inputs.drop('Sex', axis='columns', inplace=True)
print(inputs.head())

# Checking the NULL values:
print(inputs.columns[inputs.isna().any()])
print(inputs.Age[0:10])

# Filling the Na values:
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head(6))

# Training and Testing the dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

# Naive Bayes Algorithms:
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# traning the model:
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.predict(X_test[0:10]))