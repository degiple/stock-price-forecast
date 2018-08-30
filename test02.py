
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # データ準備
# ## 日経225連動型上場投資信託 1321

# In[3]:


import os

# file一覧を作成
os.chdir('C:/Users/localadmin/Documents/data/1321')
file_list = os.listdir() 

# ファイル一覧を読み込みひとつのＤａｔａＦｒａｍｅにする
data1321 = pd.DataFrame([])
for i in file_list :
    tmp = pd.read_csv(i,encoding='shift-jis', skiprows=1, index_col='日付', parse_dates=True)
    #data1321 = pd.concat([data1321,tmp],ignore_index=True)
    data1321 = pd.concat([data1321,tmp])
    
data1321.tail()
#data1321.dtypes


# ## 外国為替

# In[4]:


import os

# file一覧を作成
os.chdir('C:/Users/localadmin/Documents/data/rate_exchange')
file_list = os.listdir() 

# ファイル一覧を読み込みひとつのＤａｔａＦｒａｍｅにする
exchange = pd.DataFrame([])
for i in file_list :
    tmp = pd.read_csv(i,encoding='shift-jis',  index_col='日付', parse_dates=True)
    exchange = pd.concat([exchange,tmp])

#exchange.round().astype(int)
#exchange.info()

exchange.tail()


# ## 結合

# In[8]:


target = pd.merge(data1321, exchange, right_index=True, left_index=True, how='outer')

#NaNを削除(行列の両方)
target.dropna().dropna(axis=1) 


# In[6]:


target.info()

