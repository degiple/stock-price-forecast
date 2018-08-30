
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import os
os.chdir('C:/Users/localadmin/Documents/data/1321')


# In[4]:


file_list = os.listdir() # file一覧を作成


# In[5]:


# ファイル一覧を読み込みひとつのＤａｔａＦｒａｍｅにする
target = pd.DataFrame([])
for i in file_list :
    tmp = pd.read_csv(i,encoding='shift-jis',skiprows=1)
    target = pd.concat([target,tmp],ignore_index=True)


# In[6]:


target.head()


# In[7]:


target.tail()


# In[8]:


target.info()


# In[9]:


target.describe()


# In[10]:


# 文字列になっている日付を日付型データに変換してindexに設定
target.index = pd.to_datetime(target['日付'])


# In[11]:


# ローソクチャートを作成するためのライブラリをimport
# import matplotlib.finance as mpf
import mpl_finance as mtf


# In[12]:


# 作図エリアの設定
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,1,1)

# 作図対象データを抽出
tmp = target['2001-7-13':'2017-12-29']

# 描画関数の実行
mtf.candlestick2_ochl(opens=tmp['始値'],
                      closes=tmp['終値'],
                      highs=tmp['高値'],
                      lows=tmp['安値'],
                      ax=ax1,width=0.8,
                      )

# X軸のラベルの設定
labels = np.array([str(x.month)+'／'+str(x.day) for x in tmp.index])  #月と日のデータを取り出して編集
tick = np.arange(0,len(tmp),10)                                       #ラベル10日刻みで出力するためのインデックスを作成
ax1.set_xticks(tick)                                                  #10日刻みにラベルを設定
ax1.set_xticklabels(labels[tick])                                     #上記の設定内容に文字列を付与

plt.show()


# In[13]:


plt.figure(figsize=(20,5))

# 線グラフの描画
plt.plot(target['始値'],label='始値')
plt.plot(target['終値'],label='終値')
# 凡例の出力
plt.legend()

plt.show()


# In[43]:


# 陰線と陽線の比率を可視化する

# 始値と終値からＣｌａｓｓを設定する
target['Class'] = 'Even'
target.loc[target['終値']>target['始値'],'Class'] = 'up'
target.loc[target['終値']<target['始値'],'Class'] = 'down'


# In[44]:


# 年をキーとして設定する
target['year'] = [x.year for x in target.index]


# In[45]:


# 各年のクラスを可視化
# sns.set()
plt.figure(figsize=(15,5))
sns.countplot(x='year',data=target,hue='Class',)
plt.show()


# In[48]:


# 始値と終値の差異を時系列表示

tmp = target.copy()[['始値','終値']]
tmp['dif'] = tmp['終値']-tmp['始値']


# In[47]:


plt.figure(figsize=(20,5))
plt.plot(tmp['dif'])
plt.show()


# In[18]:


# 差異を月別平均に変換

tmp2 = tmp.resample('M').mean()['dif']

plt.figure(figsize=(20,5))
plt.plot(tmp2)
plt.show()


# In[19]:


# 差異を22日（おおよそ1ヶ月）の移動平均に変換

tmp2 = tmp.rolling(window=22).mean()['dif']

plt.figure(figsize=(20,5))
plt.plot(tmp2)

# 基準線を描画する
plt.axhline(0,c='r')
plt.show()


# In[20]:


# 始値と終値の差異を比率にする
tmp['dif_pct'] = (tmp['終値'] / tmp['始値']) -1


# In[21]:


# 差異の比率を描画
plt.figure(figsize=(20,5))
tmp2 = tmp['dif_pct']
plt.plot(tmp2)
plt.axhline(0,c='r')
plt.show()


# In[22]:


# 比率の分布の可視化

plt.hist(tmp['dif_pct'],bins=80)
plt.show()


# In[23]:


# 変化率の統計量を算出
tmp['dif_pct'].describe()


# In[24]:


# 始値と終値について対前日比を計算して分布を可視化

plt.figure(figsize=(20,5))
sns.distplot(tmp['始値'].pct_change().dropna(),label='始値')
sns.distplot(tmp['終値'].pct_change().dropna(),label='終値')
plt.legend()
plt.show()


# In[25]:


# 自己相関の分析に必要なライブラリをimport
import statsmodels.api as sm


# In[26]:


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
# 自己相関（t期の値がt+n期の値とどの程度相関しているかを示す）
_ = sm.tsa.graphics.plot_acf(tmp['dif_pct'],lags=60,ax=ax1)
# 偏自己相関（他の期の相関を除去して真の相関を示す）
_ = sm.tsa.graphics.plot_pacf(tmp['dif_pct'],lags=60,ax=ax2)


# # 高値と安値の可視化

# In[27]:


# 始値と終値の差異の分布を表す関数に設定するパラメータを計算する。

MEAN = np.mean(tmp['dif_pct'])   #平均
STD = np.std(tmp['dif_pct'])     #標準偏差

tmp = target.copy()[['安値','高値','始値']]
tmp['高値率'] = (tmp['高値'] / tmp['始値']) -1
tmp['安値率'] = (tmp['安値'] / tmp['始値']) -1

# 正規分布に拠った確率値を計算するライブラリのimport

import scipy.stats as stats

lower = stats.norm.cdf(x=tmp['安値率'],loc=MEAN,scale=STD)  # 安値以下を取る確率
higher = stats.norm.sf(x=tmp['高値率'],loc=MEAN,scale=STD)  # 高値以上を取る確率
tmp2 = 1.0 - (lower+higher)  #高値から安値の範囲を推移する確率

# 結果をヒストグラムにする
plt.hist(tmp2,50)
plt.show()


# # 機械学習で株価を予測（階差系列のみ）

# In[106]:


tmp_dif = tmp.drop("diff", axis=1)


# In[97]:


from sklearn.ensemble import RandomForestClassifier

tmp.drop("dif", axis=1)

# 訓練データ
df_train = tmp.iloc[1:len(tmp)-1]
df_train.tail()


# In[61]:


df_test = tmp[len(tmp)-1:len(tmp)]
df_test


# In[93]:


y_train = []

for s in range(0, len(df_train)):

    if df_train['diff'].iloc[s] > 0:
        y_train.append(1)
    else:
        y_train.append(-1)


# In[94]:


len(y_train)


# In[95]:


rf = RandomForestClassifier(n_estimators=len(df_train), random_state=0)
rf.fit(df_train, y_train)


# In[96]:


rf.predict(df_test)

