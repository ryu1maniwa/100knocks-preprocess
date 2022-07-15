## 環境構築
WSL2上のUbuntu 20.04LTSでデータサイエンス100本ノック  
https://www.aise.ics.saitama-u.ac.jp/~gotoh/DS100KnocksOnUbuntu2004InWSL2.html
## Python
### 2022/07/11 Problem1~18まで
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

### 2022/07/12 Problem19~39まで
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
`groupby('store_cd').product_cd.apply(関数).reset_index()`  
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
### 2022/07/13 Problem40~70まで
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
### 2022/07/14 Problem71~80まで
- 経過時間を計算する：`relativedelta(x1, x2)`
- 日付をエポック秒に変換：`datetime.strftime('%s')`
- 日付の曜日を月曜日からの経過日数として取得：`datetime.weekday()`
- 無作為抽出法：`df.sample(frac=0.01)`
- 無作為抽出法では試行によってはサブセットに偏りができる場合もあった  
この問題を解決するのが層化抽出法
```
# アンダースコアは慣例的に、必要のない値の代入先として使用される
_, df_tmp = train_test_split(df_customer, test_size=0.1, 
                             stratify=df_customer['gender_cd'])
```
- preprocessing.scale()を用いることで、平均から3σを超えて離れた外れ値を抽出しやすくなる
- pd.dataframe.query('')内で変数にアクセスしたいとき：'@変数名'
```
df_tmp.query('amount < @amount_low or @amount_hight < amount')
```
### 2022/07/15 Problem81~90まで
- 条件を満たすデータをマスクし、値を代入する
```
pd.dataframe.mask(条件, 値)
```
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