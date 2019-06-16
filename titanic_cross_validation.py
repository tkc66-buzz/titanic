import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score

# 訓練データのcsvファイルの読み込み
train = pd.read_csv("relevant_file/train.csv").replace("male", 0).replace("female", 1)

# 年齢の中央値で欠損値を補完
train["Age"].fillna(train.Age.median(), inplace=True)

# SibSp：兄弟、配偶者の数、
# Parch：両親、子供の数に自分(+1)として家族の人数という変数とした。
# 理由としては、5人以上の家族は生存確率が低かったため、それが有意か確かめるため。
# また、不要な変数を削除した。
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
new_train = train.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

train_data = new_train.values
x = train_data[:, 2:]  # データとして必要なのはPclass以降の変数
y = train_data[:, 1]  # 正解データ
print(x.shape)
print(y.shape)

# 5分割での交差検証
logreg = LogisticRegression()
scores = cross_val_score(logreg, x, y, cv=5)
# 各分割におけるスコア
print('Cross-Validation scores: {}'.format(scores))
# スコアの平均値
print('Average score: {}'.format(np.mean(scores)))
