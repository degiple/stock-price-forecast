# 株価予測プログラムの使い方

## ０．プログラム動作環境  

- Anaconda Navigator 1.8.7
- jupyter notebook 5.5
- Python 3.6.5
- pandas 0.23

## １．学習モデルを作成するプログラム（create_forcast-model.ipynb）  

### １－１．最新データ準備・配置（必要があれば）

#### 1321 日経225連動型上場投資信託

以下URL先より、2018年株価データ(1321_2018.csv)をダウンロードする。
[個別株価データ](https://kabuoji3.com/stock/1321/2018/)

1321_2018.csv を ~/program/data/1321 に配置する。  
※上書き保存OK

#### 1321 日経225連動型上場投資信託

以下URLより、為替データをダウンロードする。
[外国為替公示相場ヒストリカルデータ(日次)](https://www.mizuhobank.co.jp/market/csv/quote.csv)

### １－２．プログラム実行

Anaconda Navigator / jupyter notebook を起動する

jupyter notebook 上から
~/program/create_forcast-model.ipynb
を実行する。

メニュー「kernel」より
restart & Run ALL
を実行する。

### １－３．保存された学習モデルの確認  
