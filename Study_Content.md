# 環境構築
WSL2上のUbuntu 20.04LTSでデータサイエンス100本ノック  
https://www.aise.ics.saitama-u.ac.jp/~gotoh/DS100KnocksOnUbuntu2004InWSL2.html
# データベースの基礎
## E-R図
## デ－タベースの正規化
- データの重複をなくし整合的にデータを取り扱えるようにデータベースを設計すること。正規化をすることで、データの追加・更新・削除などに伴うデータの不整合や喪失が起きるのを防ぎ、メンテナンスの効率を高めることができる
- 正規化の段階には、第1～第5正規形およびボイスコッド正規形がある
### 非正規形
- 正規化がまったく行われておらず、1行の中に複数の繰り返し項目が存在するようなテーブルのこと
- リレーショナルデータベース(RDB)では、レコード単位で個々のデータを扱うため、非正規形のデータはデータベースに格納できない
### 第1正規形
- 非正規形から第1正規形への正規化
    - 非正規形の繰り返し項目を別レコードとして独立させたもの
    - 他のカラムから導出可能な項目を削除しておく
- データベースに格納できる
- 独立した情報をすべて同一のレコードで扱っているため、データ管理の観点からは不十分
### 第2正規形
- 第1正規形から第2正規形への正規化
    - レコードを一意に定める要素を主キー(PK)と呼ぶ
    - いずれのテーブルにおいても非キー属性が主キーに従属するように、データを別テーブルに分離することで第2正規形に正規化できる
- 一つの変更に対して複数レコードを更新する必要があるため非効率
### 第3正規形
- 依存関係のあるデータをわけることでデータの変更を複数でする必要を避ける
- 第2正規形から第3正規形への正規化
    - 主キー以外の項目について項目同士で依存関係を持っているもの（推移的関数従属と言う）も、別テーブルに切り分ける
- 検索効率を考えて、あえて正規化の程度を落とすこともある
- アプリケーションの利用シーンやパフォーマンス要件などに応じて柔軟にデータベースを設計したい
# Python
## モジュール、パッケージ、ライブラリの説明
- Pythonモジュールとは、Pythonのコードをまとめたファイル
- パッケージとは、複数のモジュールがフォルダに集まったもの
- ライブラリとは、フォルダやファイルが集まったもの
- サイドパーティ・ライブラリはpip（The Python Package Installer）を用いてインストールする
```
$ pip install --upgrade pip
$ pip install numpy
$ pip install pandas
$ pip install scikit-learn
$ pip install python-dateutil
$ pip install psycopg2
$ pip install sqlalchemy
```
## それぞれのモジュールの説明
```
import os
```
- OSの機能を利用するためのモジュール。主にファイルやディレクトリ操作が可能で、ファイルの一覧やpathを取得できたり、新規にファイル・ディレクトリを作成することができる  
参考：https://www.sejuku.net/blog/67787  
- Operating System(OS)：ユーザーとハードウェア、応用ソフトウェアとハードウェアをつなぐもの
```
import pandas as pd
```
- Pandas(パンダス)とは、データ解析のためのサイドパーティ・ライブラリ。データ分析作業を支援するためのモジュールが含まれており、表形式のデータをSQLまたはRのように操作するために用いる  
参考：https://utokyo-ipp.github.io/7/7-1.html
```
import numpy as np
```
- 多次元配列を効率的に扱うための外部パッケージ。ベクトルや行列の演算を効率的にすることができる
```
from datetime import datetime, date
```
- 標準ライブラリのdatetimeモジュールからdatetime, dateというオブジェクトをインポートする  
参考：https://docs.python.org/3/library/datetime.html#datetime-objects
```
from dateutil.relativedelta import relativedelta
```
- dateutil：標準のdatetimeモジュールに強力な拡張機能を提供するモジュール  
参考：https://dateutil.readthedocs.io/en/stable/
```
import math
```
- 数学的な計算をするのに役立つ標準モジュール
```
import psycopg2
```
- PythonからPostgreSQLへアクセスするための外部ライブラリ。SQL文を実行することが可能   
参考：https://resanaplaza.com/2021/09/08/%E3%80%90-python-%E3%80%91psycopg2%E3%81%A7postgresql%E3%81%AB%E3%82%A2%E3%82%AF%E3%82%BB%E3%82%B9%E3%81%97%E3%82%88%E3%81%86%EF%BC%81/  
DBからデータを読み取り、pythonで分析したい場合は以下のようにする
```
pgconfig = {
    'host': 'db',
    'port': os.environ['PG_PORT'],
    'database': os.environ['PG_DATABASE'],
    'user': os.environ['PG_USER'],
    'password': os.environ['PG_PASSWORD'],
}

# pd.read_sql用のコネクタ
conn = psycopg2.connect(**pgconfig)

df_customer = pd.read_sql(sql='select * from customer', con=conn)
df_category = pd.read_sql(sql='select * from category', con=conn)
df_product = pd.read_sql(sql='select * from product', con=conn)
df_receipt = pd.read_sql(sql='select * from receipt', con=conn)
df_store = pd.read_sql(sql='select * from store', con=conn)
df_geocode = pd.read_sql(sql='select * from geocode', con=conn)
```
```
from sqlalchemy import create_engine
```
- ORM：sqlalchemyをインポートする  
Object Relational Mapper(ORM)とは、テーブルとクラスを1対1に対応させて、そのクラスのメソッド経由でデータを取得したり、変更したりできるようにする存在。
- ORMを使用することで
    - 複数のDBを併用する場合やDBを変更する場合にも、コードの書き換えの必要がなくなる
    - SQLAlchemyを使うとSQLを直接記述することなしに、DBを"Pythonic"に操作できる  
参考：https://qiita.com/arkuchy/items/75799665acd09520bed2
```
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
```
- scikit-learn：様々な機械学習の手法が統一的なインターフェースで利用できる外部ライブラリ。  
ndarrayでデータやパラメータを取り扱うため、他のライブラリとの連携もしやすい  
参考：https://tutorials.chainer.org/ja/09_Introduction_to_Scikit-learn.html
- scikit-learnの一部モジュールの説明
    - preprocessing：データの前処理に用いる
    - SimpleImputer：欠損値の補間に用いる
    - train_test_split：データの分割に用いる
    - TimeSeriesSplit：時系列データを用いた交差検証をするために用いる
    - RandomUnderSampler：不均衡なデータのアンダーサンプリングに用いる
## 2022/07/11 Problem1~18まで
- データの列を指定して抽出
- 項目名を変更しながら抽出：`pd.dataframe.rename(columns='':'')`  
参考：https://note.nkmk.me/python-pandas-dataframe-rename/
- 条件を満たすデータを抽出：`pd.dataframe.query('条件', engine='python')`
- 指定した正規表現のパターンに一致するか  
条件 = `pd.dataframe.str.match(r'文字列')`
- SQLの正規表現のパターンは以下のサイトを参照  
参考：https://qiita.com/syuki-read/items/6c5f139abd676fcc0ca2
- rまたはRを文字列リテラルと呼ぶ。  '...'や"..."の前につけるとエスケープシーケンスを展開せず、そのままの値が文字列となる。このような文字列はraw文字列（raw strings, r-strings）と呼ばれる。
- データフレームのソート：`pd.dataframe.sort_values('列', ascending=True)`  
参考：https://note.nkmk.me/python-pandas-sort-values-sort-index/

## 2022/07/12 Problem19~39まで
- データの件数のカウント：`len(dataframe)`
- ユニーク件数のカウント：`pd.dataframe.nunique()`  
参考：https://note.nkmk.me/python-pandas-value-counts/
- 同じ値を持つデータをまとめて、それぞれに対して共通の操作を行いたい時
`pd.dataframe.groupby('ラベル')`  
参考：https://qiita.com/propella/items/a9a32b878c77222630ae
- グルーピングした後、更にデータを算出したい場合  
`groupby('ラベル').agg({カラム：集約関数}).reset_index()`  
※aggはAggregationの略で「集約」を意味する  
グルーピングすることで行に欠番が出てしまうのでインデックスを振りなおす  
カラムごとに異なる統計量を自由に算出することが出来る  
集約関数には自作関数を適用することができる  
参考：https://vector-ium.com/pandas-agg/  
　　　https://note.nkmk.me/python-pandas-reset-index/
- apply()とは、DataFrame, Series型に備わっているメソッドの一つで、groupbyされたDataFrameにおける各々のvalueを引数として、apply()の引数に与えられた関数のreturn値を返すことができる。  
`groupby('store_cd').product_cd.apply(関数, axis=1).reset_index()`  
複数列にまたがる自作関数はaxis=1とする必要があることに注意  
参考：https://qiita.com/hisato-kawaji/items/0c66969343a196a65cee
- 無名関数（ラムダ式）：`変数 = lambda 引数1, 引数2 : 式`  
`apply(lambda x: x.mode())` のようにも使える
- Numpyはデフォルト値で標本分散が求まり、pandasでは不偏分散の値が求まるようになっている。紛らわしいので指定した方が分かりやすい。  
標本分散を求める：`std(ddof=0)`
不偏分散を求める：`std(ddof=1)`  
参考：https://itstudio.co/2021/03/19/11249/
- pandas.DataFrameの任意の位置のデータを取り出したり変更（代入）したりするには、at, iat, loc, ilocを使う  
`df_receipt.amount.describe().loc['25%':'max']`  
参考：https://note.nkmk.me/python-pandas-at-iat-loc-iloc/
- パーセンタイルは、特定の値を下回るスコアの割合を示す  
パーセンタイルの計算：`percentile(a, q)`  
※aは配列、qはパーセンタイルの計算数を表す  
参考：https://www.delftstack.com/ja/howto/python/python-percentile/
- Dataframeの統合：`merge(df1, df2, how='', on='')`  
結合の仕方を指定できる：how='inner', 'left', 'right', 'outer'  
デフォルトでは2つのpandas.DataFrameに共通する列名の列をキーとして結合処理が行われるが、明示的に指定する場合は引数onを使う。省略できるが明示しておいたほうが分かりやすい  
参考：https://note.nkmk.me/python-pandas-merge-join/
- 結合の仕方
    - 内部結合：2つのテーブルの合体可能なデータのみ取り出す
    - 左外部結合：テーブル1の全データを取り出して、それにテーブル2のデータをくっつける
    - 右外部結合：テーブル2の全データを取り出して、それにテーブル1のデータをくっつける
    - 完全外部結合：両方のテーブルの全データを取り出して、くっつけられる範囲でくっつける
    - 直積結合：両方のテーブルの全データを取り出して、すべての組み合わせでくっつける  
    参考：https://wa3.i-3-i.info/word15315.html
- 欠損値の扱い
    - NaNを削除：`dropna()`
    - NaNを置換（列ごと）：`fillna({key:value})`
    - NaNを補間：`interpolate()`
    - NaNをカウント：`isnull().sum()`  
参考：https://note.nkmk.me/python-pandas-nan-dropna-fillna/
- 重複した行の抽出：`pd.dataframe.duplicated()`  
引数subsetで重複を判定する列を指定できる  
データフレームから重複しているデータを削除する↓  
`df_data[~df_data.duplicated(subset=["", ""])`  
参考：https://note.nkmk.me/python-pandas-duplicated-drop-duplicates/
## 2022/07/13 Problem40~70まで
- pythonにはcross join(直積結合)の機能がないため、cross join用のキーを新たに作成して完全外部結合することで疑似的に直積結合できる
```
df_store_tmp['key'] = 0 # cross join用のキー
df_product_tmp['key'] = 0 # cross join用のキー
# outer joinだがキーが全て一緒なのでcross joinになる
len(pd.merge(df_store_tmp, df_product_tmp, how='outer', on='key'))
```
- データを行・列方向にずらす：`pd.dataframe.shift()`  
参考：https://note.nkmk.me/python-pandas-shift/
- フォーマット済み文字列リテラル(f文字列)：
`f"xxxx {値:書式指定子} xxxx"`  
参考：https://www.javadrive.jp/python/string/index25.html
- データ型の変換：`pd.dataframe.astype('型')`  
参考：https://note.nkmk.me/python-pandas-dtype-astype/
- 小数点以下の扱い  
`pd.dataframe.apply(lambda x: np.floor(x))`
    - 切り捨て：`np.floor()`
    - 丸め込み：`np.round()`
    - 切り上げ：`np.ceil()`  
    math.floor()などもあるが、欠損値の扱いができないのが欠点。
- データの整形には、stack, unstack, pivotが用いられる
    - 列から行へピボット: stack()
    - 行から列へピボット: unstack()
    - 行と列を指定してピボット（再形成）: pivot()  
    参考：https://note.nkmk.me/python-pandas-stack-unstack-pivot/
- ピボットテーブル：
`pd.pivot_table(data, index=, columns=, values=, aggfunc=)`  
を用いることで、カテゴリデータをカテゴリごとにグルーピングし、それぞれの統計量を確認・分析できる
    - data：用いるデータの指定
    - index: グルーピングしたい行を指定
    - columns: グルーピングしたい列を指定
    - values：統計量を算出したい列名の指定
    - aggfunc：統計量を算出する関数を指定  
参考：https://note.nkmk.me/python-pandas-pivot-table/
- 既存の列をインデックス（行名、行ラベル）に割り当てる：`set_index('column')`
- 文字列を置換する：`replace({'前':'後'})`
- 行名・列名を変更する：`rename(columns={'前':'後'})`
- 文字列をdatetime64[ns]型に変換する：`pd.to_datetime()`  
数値型の場合、astype(str)で文字列に変換してから使う必要がある  
UNIX秒からの変換の場合、引数としてunit='s'を指定  
参考：https://deepage.net/features/pandas-to-datetime.html
- datetime型からformatの文字列を返す：`Series.dt.strftime(format)`
- datetime型から年を返す：`Series.dt.year`  
他のパターン：https://qiita.com/Takemura-T/items/79b16313e45576bb6492
- Seriesは一次元のデータ構造、DataFrameは二次元のデータ構造で、DataFrameはSeriesから構成される
- 場合分けを適用：`apply(lambda x: 1 if x>2000 else 0)`
- 先頭3桁が100〜209のものを1、それ以外のものを0に二値化
`np.where(df_tmp['postal_cd'].str[0:3].astype(int).between(100, 209), 1, 0)`
- map()の引数に辞書型を指定すると、keyと一致する要素がvalueに置き換えられる  
`str[0:3].map({'埼玉県':'11', '千葉県':'12'})`
- ダミー変数化：`pd.get_dummies(data, columns=list)`  
引数columnsにダミー化したい列の列名をリストで指定
- データの前処理：`from sklearn import preprocessing`
    - 平均0、標準偏差1に標準化：`preprocessing.scale(data)`
    - 最小値0、最大値1に正規化：`preprocessing.minmax_scale`  
    参考：https://qiita.com/Umaremin/items/fbbbc6df11f78532932d
- 対数化
    - 常用対数化（底10）：`np.log10(data + 1)`
    - 自然対数化（底e）：`np.log(data + 1)`  
対数化では真数条件に引っかからないように1を加えている
- 重複したデータを削除：`pd.dataframe.drop_duplicates()`
## 2022/07/14 Problem71~80まで
- 経過時間を計算する：`relativedelta(x1, x2)`
- 日付をエポック秒に変換：`datetime.strftime('%s')`
- 日付の曜日を月曜日からの経過日数として取得：`datetime.weekday()`  
参考：https://kino-code.com/python-datetime-weekday/
- 無作為抽出法：`df.sample(frac=0.01)`
- 無作為抽出法では試行によってはサブセットに偏りができる場合もあった  
この問題を解決するのが層化抽出法
```
_, df_tmp = train_test_split(df_customer, test_size=0.1, 
                             stratify=df_customer['gender_cd'])
```
アンダースコアは慣例的に、必要のない値の代入先として使用される  
参考：https://mako-note.com/ja/python-underscore/
- preprocessing.scale()を用いることで、平均から3σを超えて離れた外れ値を抽出しやすくなる
- pd.dataframe.query('')内で変数にアクセスしたいとき：'@変数名'
```
df_tmp.query('amount < @amount_low or @amount_hight < amount')
```
## 2022/07/15 Problem81~90まで
- 条件を満たすデータをマスクし、値を代入する
```
pd.dataframe.mask(条件, 値)
```
参考：https://kino-code.com/python-pandas-mask/
- apply()はfor文処理で遅いため、できるならnumpyを使って計算させたい
- 値が最も大きいものを残して重複を削除  
→値が大きい順にソートしてから、上のデータを残して重複を削除
```
df_customer_tmp.sort_values(['customer_id', 'amount'], \
                            ascending=[False, True], inplace=True)

df_customer_tmp.drop_duplicates(
                subset=['customer_name', 'postal_cd'], 
                keep='first', inplace=True)
```
- 8:2の割合でランダムに学習用データとテスト用データに分割
```
df_train, df_test = train_test_split(df_customer_tmp, test_size=0.2)
```
## 2022/07/16 Problem91~100まで
- 条件に応じて代入する値を変える：np.where()
```
np.where(条件, 0, 1)
```
- 不均衡なデータを１：１にアンダーサンプリングする：RandomUnderSampler
```
# is_by_flagを基準にdf_tmpをアンダーサンプリング
rs = RandomUnderSampler(random_state=71)
df_down_sampling, _ = rs.fit_resample(df_tmp, df_tmp['is_by_flag'])
```
- csvファイルへ出力：pd.to_csv
```
# コード例2（BOM付きでExcelの文字化けを防ぐ）
df_product_full.to_csv('data/P_df_product_full_UTF-8BOM_header.csv', 
                encoding='utf-8-sig', header=True, index=False)
```
- csvファイルを読み込む：pd.read_csv
```
df_product_full_1 = \
        pd.read_csv('data/P_df_product_full_UTF-8BOM_header.csv')
```