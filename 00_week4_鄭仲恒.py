# 總分為 210 分
# 及格分數為 90 分

# %%
#--------------- 簡答題 -------------------
# 問題（簡答）：請查看 training1.JPG，並請說明本圖之管理意涵與折線意涵
# 15分
#圖片意涵可以建兩種模型Xgboost與RandomForest，並且比較兩者的表現，可以看到RandomFores的表現比RandomForest好，因此可以選擇RandomForest模型
#頂部的隨機森林模型在測試集上達到0.868的準確率。下半部分顯示另一個模型XGBoost,其在測試集上的準確率為0.856,稍低於隨機森林模型。
#圖中繪製了這兩個模型在不同參數設置下的預測得分曲線。可以看出,隨機森林模型的得分曲線相對平滑,而XGBoost的得分曲線則有一個明顯的峰值,表明它對某些特定參數值的變化更為敏感。

# %%
# 問題（簡答）：請查看 training2.png，並請說明本圖之管理意涵
# 15分
#這個圖像是一個決策樹，用於根據一系列的規則來做出預測。從最頂端的節點開始，根據年齡是否小於等於41.5歲來分類數據。左邊的分支是註冊使用時間小於等於2.5的情況，右邊則是根據是否為活躍會員進行進一步的分類。每個節點顯示了熵值、樣本數量和類別分佈，葉節點給出最終的預測結果。熵值衡量節點的不純度，樣本數表示該節點的數據量，而value則展示了每個節點的類別分佈情況。

# %%
# 問題（簡答）：請問什麼是 Accuracy trap ？
# 15分
#Accuracy trap是指在機器學習中，當樣本不平衡時，模型的準確率會高，但是對於少數類別的預測效果不好，這種情況就是準確率陷阱。


# %%
# 問題（簡答）：請說明 p-value 與信賴區間是什麼？
# 15分
# p-value 是統計學中的一個概念，是用來評估樣本數據對於虛無假設的支持程度。當p-value小於顯著水準時，我們就可以拒絕虛無假設，認為樣本數據支持對立假設。
# 信賴區間是用來估計母體參數的一個區間，通常用來估計平均值或者變異數等。

# %%
# 問題（簡答加分題）：請問進行機器學習時，如何判斷 error 是從 variance 而來還是 bias 而來？
# 若是從 variance 而來，該如何解決？
# 若是從 bias 而來，該如何解決？
# 該如何解決？
# 40分
#當模型的bias很高時，說明模型對於訓練集的擬合不夠，這時可以嘗試增加模型的複雜度，比如增加模型的參數，或者使用更複雜的模型。
#當模型的variance很高時，說明模型對於訓練集的擬合過度，這時可以嘗試減少模型的複雜度，比如減少模型的參數，或者使用更簡單的模型。



# 至此總分100


# %%
#--------------- 機器學習 實作題 -------------------
# 問題：請看 bank.pdf 檔案，並依照過往所學之機器學習知識，對 bank_train.csv 與 bank_test.csv 執行分析
# 也可以參考 pdf 裡面的 Tip1 與 Tip2
# 60分

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('bank_train.csv')
df
df.head()
df.info()
#%%
print('不購買', round(df['buy'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('購買', round(df['buy'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


#%%
def get_dummies(dummy, dataset):
    ''''
    make variables dummies
    ref：http://blog.csdn.net/weiwei9363/article/details/78255210
    '''
    dummy_fields = list(dummy)
    for each in dummy_fields:
        dummies = pd.get_dummies( dataset.loc[:, each], prefix=each ) 
        dataset = pd.concat( [dataset, dummies], axis = 1 )
    
    fields_to_drop = dummy_fields
    dataset = dataset.drop( fields_to_drop, axis = 1 )
    return dataset

df = get_dummies(['UID', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'], df)


#%%
df.info()
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


X = df.drop(['buy'], axis=1)
y = df['buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
train_uid = X_train['UID']
test_uid = X_test['UID']
del X_train['UID']
del X_test['UID']
#%%
from sklearn.linear_model import LogisticRegression 
logistic_reg =LogisticRegression()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
logistic_reg.fit(X_train, y_train)  
y_pred = logistic_reg.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
logistic_pred_proba = logistic_reg.predict_proba(X_test) 
print('Classification Report: \n', classification_report(y_test, y_pred))


#%%

from sklearn.preprocessing import StandardScaler    
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
print('Classification Report: \n', classification_report(y_test, y_pred))

# 至此總分170

# %%
#---------------Streamlit  實作題 -------------------
# 問題：請查看 training3.JPG 配合 diabetes_prediction_ml_app 資料夾
# 請將紅框處轉變成「箭頭」所指之處（更改功能）
# 請將藍框處轉變成「箭頭」所指之處（更改功能）
# 請將完成的檔案，上傳到 streamlit cloud(https://share.streamlit.io/)，並在此附上網址
# 請注意 streamlit cloud 的Python版本限制在 <= 3.10
# 40分





# 至此總分210
