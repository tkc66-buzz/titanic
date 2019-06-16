import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("./relevant_file/train.csv").replace("male", 0).replace("female", 1)

df["Age"].fillna(df.Age.median(), inplace=True)

split_data = []
for survived in [0, 1]:
    split_data.append(df[df.Survived == survived])

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)

temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

train_data = df2.values
xs = train_data[:, 2:]  # Pclass以降の変数
y = train_data[:, 1]  # 正解データ

forest = RandomForestClassifier(n_estimators=100)

# 学習
forest = forest.fit(xs, y)

test_df = pd.read_csv("./relevant_file/test.csv").replace("male", 0).replace("female", 1)
# 欠損値の補完
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

test_data = test_df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

print(len(test_data[:, 0]), len(output))
zip_data = zip(test_data[:, 0].astype(int), output.astype(int))
predict_data = list(zip_data)

with open("./relevant_file/predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:, 0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
