ブログ記事「The Annotated Transformer」のコードです。

[日本語版ノートブック](AnnotatedTransformer.ja.ipynb)

http://nlp.seas.harvard.edu/annotated-transformer/

[![原文をColabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)

![image](https://user-images.githubusercontent.com/35882/166251887-9da909a9-660b-45a9-ae72-0aae89fb38d4.png)

# パッケージの依存関係

[uv](https://docs.astral.sh/uv/) で依存パッケージをインストールします。Python 3.15と `uv.lock` に記載されたバージョンを使う仮想環境が作成されます。

```sh
uv sync
source .venv/bin/activate
```

依存パッケージにはPyTorch 2.15が含まれます。Linux x86-64向けパッケージにはCUDA 13.2ライブラリが含まれ、CUDAの使用には対応するNVIDIA GPUとドライバーが必要です。CPUでも実行できます。`uv sync` は `pyproject.toml` に定義された依存関係を使います。開発が終了したtorchtextの語彙処理はローカル実装に置き換え、既存の語彙キャッシュは新形式で作り直します。

# ノートブックの実行

`uv run jupyter notebook` でJupyterを起動し、`AnnotatedTransformer.ipynb` または `AnnotatedTransformer.ja.ipynb` を開いて、セルを順番に実行します。初回は独英対訳データセットMulti30kを `.data/multi30k` にダウンロードし、次回以降はキャッシュを利用します。トークン化にはspaCyの組み込みトークナイザを使うため、学習済みのspaCyモデルは不要です。

デフォルトでは、CPUとGPUのどちらでも試せる軽量なデモとして、小さなモデルを64文対で1エポック学習します。翻訳結果は処理の流れを示すためのもので、論文の結果とは比較できません。元の大きなモデルとデータセット全体で8エポック学習する場合は、Jupyterの起動前にモードを指定します。

```sh
ANNOTATED_TRANSFORMER_MODE=full uv run jupyter notebook
```

完全な学習には大幅に多くの時間とメモリが必要で、CUDA対応GPUを推奨します。設定が異なるチェックポイントは別々に保存されます。
GPUが利用可能な環境でもCPUを使う場合は、`ANNOTATED_TRANSFORMER_DEVICE=cpu uv run jupyter notebook` でJupyterを起動してください。

# ノートブックのセットアップ

The Annotated Transformerは [jupytext](https://github.com/mwouts/jupytext) を使って作成されています。

通常のノートブックでは、セルの出力がリポジトリの履歴に残るため、コミット間の差分を確認しにくくなります。jupytextを使うと、Pythonスクリプト（`.py` ファイル）とノートブックファイルをプラグインによって自動的に同期できます。

リポジトリにコミットされたPythonスクリプトにはすべてのセルの内容が含まれ、ノートブックファイルの生成に使えます。スクリプトは通常のPythonソースファイルで、Markdownセクションは決められたコメント形式で記述され、出力は保存されません。jupytextで生成するノートブックは生成物として扱い、Gitリポジトリにはコミットしません。

日本語版の `AnnotatedTransformer.ja.ipynb` は別途管理しており、`make notebook` では生成・更新されません。

このリポジトリを使う前に、[インストール手順](https://github.com/mwouts/jupytext/blob/main/docs/install.md)に従ってjupytextがインストールされていることを確認してください。

ソースから `.ipynb` 形式のノートブックファイルを生成するには、次を実行します（`notebook` ビルドターゲットは内部で `jupytext --to ipynb the_annotated_transformer.py` を実行します）。

```sh
make notebook
```

ノートブックのHTML版を生成するには、次を実行します。

```sh
make html
```

`make html` は、`jupytext --to ipynb the_annotated_transformer.py` でノートブックを生成し、続いて `jupyter nbconvert --to html the_annotated_transformer.ipynb` でHTMLに変換する手順をまとめたものです。

# フォーマットと静的解析

GitHub ActionsではPythonのエラーとオフラインのワークフローテストを確認しています。次のコマンドで [PEP 8のコーディング規約](https://peps.python.org/pep-0008/) に沿った書式の確認もできます。

これを手軽に実行できるように、blackとflake8を使う2つの `Makefile` ターゲットが用意されています。[black](https://github.com/psf/black#installation) と [flake8](https://flake8.pycqa.org/en/latest/) がインストールされていることを確認してください。

blackでコードを自動整形するには、次を実行します。

```sh
make black
```

または、`black --line-length 79 the_annotated_transformer.py` を直接実行します。

PEP 8違反を確認するには、次を実行します。

```sh
make flake
```

または、`flake8 --show-source the_annotated_transformer.py` を直接実行します。

プルリクエストの提出前には、`uv run python -m unittest discover -s tests -v` で学習ワークフローのテストを実行してください。
