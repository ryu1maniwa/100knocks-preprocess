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

### 2022/07/11 Problem19~39まで
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
デフォルトでは2つのpandas.DataFrameに共通する列名の列をキーとして結合処理が行われるが、明示的に指定する場合は引数onを使う。省略できるが明示しておいたほうが分かりやすい。  
参考：https://note.nkmk.me/python-pandas-merge-join/
- 欠損値の扱い
    - NaNを削除：`dropna()`
    - NaNを置換：`fillna(value)`
    - NaNを補間：`interpolate()`
    - NaNをカウント：`isnull()`  
参考：https://note.nkmk.me/python-pandas-nan-dropna-fillna/
- 重複した行の抽出：`pd.dataframe.duplicated()`  
引数subsetで重複を判定する列を指定できる  
データフレームから重複しているデータを削除する↓  
`df_data[~df_data.duplicated(subset=["", ""])`  
参考：https://note.nkmk.me/python-pandas-duplicated-drop-duplicates/