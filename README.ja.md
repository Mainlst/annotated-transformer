ブログ記事「The Annotated Transformer」のコードです。

[日本語版ノートブック](AnnotatedTransformer.ja.ipynb)

http://nlp.seas.harvard.edu/annotated-transformer/

[![原文をColabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)

![image](https://user-images.githubusercontent.com/35882/166251887-9da909a9-660b-45a9-ae72-0aae89fb38d4.png)

# パッケージの依存関係

[uv](https://docs.astral.sh/uv/) で依存パッケージをインストールします。Python 3.9と `uv.lock` に記載されたバージョンを使う仮想環境が作成されます。

```sh
uv sync
source .venv/bin/activate
```

依存パッケージには、CUDA 11.3向けにビルドされたPyTorch 1.11が含まれます。そのホイールはLinux x86-64とWindows x86-64に対応しています。CUDAを使うには対応するNVIDIA GPUとドライバーが必要です。CPUでの実行にも対応しています。`uv sync` は `pyproject.toml` に定義された依存関係を使います。

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

コードの書式を整えるため、このリポジトリではコードが [PEP 8のコーディング規約](https://www.python.org/dev/peps/pep-0008/) に準拠しているかをGitHub Actionsで確認しています。

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

プルリクエストを提出する際は、GitHub ActionsのCIでエラーにならないよう、両方のコマンドを実行してflake8のエラーを修正することを推奨します。
