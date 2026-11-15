Code for The Annotated Transformer blog post:

[日本語版 README](README.ja.md) · [日本語版ノートブック](AnnotatedTransformer.ja.ipynb)

http://nlp.seas.harvard.edu/annotated-transformer/

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)

![image](https://user-images.githubusercontent.com/35882/166251887-9da909a9-660b-45a9-ae72-0aae89fb38d4.png)




# Package Dependencies

Install the dependencies with [uv](https://docs.astral.sh/uv/). This creates
the virtual environment using Python 3.15 and the versions in `uv.lock`:

```sh
uv sync
source .venv/bin/activate
```

The locked dependencies include PyTorch 2.15. On Linux x86-64, its default
package includes CUDA 13.2 libraries; CUDA requires a compatible NVIDIA GPU
and driver. CPU execution is also supported. `pyproject.toml` defines the
dependencies used by `uv sync`. Vocabulary handling is implemented locally
because torchtext is no longer maintained. Existing vocabulary caches are
rebuilt in the new format.

# Running the notebook

Start Jupyter with `uv run jupyter notebook` and open either
`AnnotatedTransformer.ipynb` or `AnnotatedTransformer.ja.ipynb`. Run the
cells in order. The first run downloads the Multi30k German–English data into
`.data/multi30k`; subsequent runs use the local cache. The notebook uses
spaCy's built-in German and English tokenizers, so pretrained spaCy models are
not required.

By default, the translation example runs a small demonstration on either CPU
or GPU: it trains for one epoch on 64 sentence pairs with a reduced model.
Its translations illustrate the workflow and are not comparable to the paper's
results. To use the original larger model and train on the full dataset for
eight epochs, set the mode before starting Jupyter:

```sh
ANNOTATED_TRANSFORMER_MODE=full uv run jupyter notebook
```

Full training requires substantially more time and memory; a CUDA GPU is
recommended. Checkpoints for different configurations are stored separately.
To use the CPU even when a GPU is available, start Jupyter with
`ANNOTATED_TRANSFORMER_DEVICE=cpu uv run jupyter notebook`.

# Notebook Setup

The Annotated Transformer is created using [jupytext](https://github.com/mwouts/jupytext).

Regular notebooks pose problems for source control - cell outputs end up in the repo history and diffs between commits are difficult to examine. Using jupytext, there is a python script (`.py` file) that is automatically kept in sync with the notebook file by the jupytext plugin.

The python script is committed contains all the cell content and can be used to generate the notebook file. The python script is a regular python source file, markdown sections are included using a standard comment convention, and outputs are not saved. The notebook itself is treated as a build artifact and is not commited to the git repository.

Prior to using this repo, make sure jupytext is installed by following the [installation instructions here](https://github.com/mwouts/jupytext/blob/main/docs/install.md).

To produce the `.ipynb` notebook file using the markdown source, run (under the hood, the `notebook` build target simply runs `jupytext --to ipynb the_annotated_transformer.py`):

```
make notebook
```

To produce the html version of the notebook, run:

```
make html
```

`make html` is just a shortcut for for generating the notebook with `jupytext --to ipynb the_annotated_transformer.py` followed by using the jupyter nbconvert command to produce html using `jupyter nbconvert --to html the_annotated_transformer.ipynb`                             
 

# Formatting and Linting

GitHub Actions checks for Python errors and runs the offline workflow tests. The following commands can also be used to check formatting against [PEP 8](https://peps.python.org/pep-0008/).

To make this easier, there are two `Makefile` build targets to run automatic code formatting with black and flake8.

Be sure to [install black](https://github.com/psf/black#installation) and [flake8](https://flake8.pycqa.org/en/latest/).

You can then run:

```
make black
```

(or alternatively manually call black `black --line-length 79 the_annotated_transformer.py`) to format code automatically using black and:

```
make flake
```

(or manually call flake8 `flake8 --show-source the_annotated_transformer.py) to check for PEP8 violations.

Before submitting a PR, run `uv run python -m unittest discover -s tests -v` to check the training workflow.
