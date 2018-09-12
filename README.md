# 株価予測プログラムの使い方

## ０．プログラム動作環境  

- Anaconda Navigator 1.8.7
- jupyter notebook 5.5
- Python 3.6.5
- pandas 0.23

## １．学習モデルを作成する（create_forcast-model.ipynb）  

### １－１．最新データ準備・配置（必要があれば）

#### 1321 日経225連動型上場投資信託

- 以下URL先より、2018年株価データ(1321_2018.csv)をダウンロードする。
- [個別株価データ](https://kabuoji3.com/stock/1321/2018/)
- ダウンロードした 1321_2018.csv を ~/program/data/1321 に配置する。  
- ※上書き保存OK

#### 為替データ

- 以下URLより、為替データをダウンロードする。
- [外国為替公示相場ヒストリカルデータ(日次)](https://www.mizuhobank.co.jp/market/csv/quote.csv)
- ダウンロードした quote.csv を ~/program/data/rate_exchange に配置する。  
- ※上書き保存OK

### １－２．プログラム実行（学習モデル作成・保存）

- Anaconda Navigator, jupyter notebook を起動する
- jupyter notebook 上から ~/program/create_forcast-model.ipynb を実行する。
- メニュー「kernel」より「restart & Run ALL」を実行する。

### １－３．保存された学習モデルの確認

- ~/program/output_models ディレクトリに移動する。
- forecast-model_(日時).pickleが存在する事を確認する。

## ２．株価を予測する（forecast_start.ipynb）

### ２－１．学習モデルの配置（必要があれば）

- 使用したい学習モデル(~.pickle)を ~/program/load_model に配置する。
- 配置した学習モデルの名前を forecast-model.pickle に変更する。

### ２－２．予測用データ配置

#### 1321 日経225連動型上場投資信託

- 以下URL先より、2018年株価データ(1321_2018.csv)をダウンロードする。
- [個別株価データ](https://kabuoji3.com/stock/1321/2018/)
- excel で、ダウンロードした 1321_2018.csv を 開く。
- excel で、~/program/target/1321_target.csv を開く。。
- 1321_2018.csv から、予測したい日の前営業日データ(行)を、行コピーする。
- 1321_target.csv の２行目に、先ほど行コピーしたデータを貼り付け、保存する。
- excel を終了する。

#### 為替データ

- 以下URLより、為替データをダウンロードする。
- [外国為替公示相場ヒストリカルデータ(日次)](https://www.mizuhobank.co.jp/market/csv/quote.csv)
- excel で、ダウンロードした quote.csv を 開く。
- excel で、~/program/target/quote_target.csv を開く。
- quote.csv から、予測したい日の前営業日データ(行)を、行コピーする。
- quote_target.csv の２行目に、先ほど行コピーしたデータを貼り付け、保存する。
- excel を終了する。

### ２－３．プログラム実行・結果確認（株価予測）

- Anaconda Navigator, jupyter notebook を起動する
- jupyter notebook 上から ~/program/forecast_start.ipynb を実行する。
- メニュー「kernel」より「restart & Run ALL」を実行する。
- 予測結果について確認する。（UP or DOWN）
