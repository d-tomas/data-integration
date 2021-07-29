# Contextual word embeddings for tabular data search and integration

This code allows computing table similarity based on differnt contextual and non-contextual word embedding models. This similarity can be further used for *ad hoc* table retrieval and integration (*join* and *union* operations).

## Getting Started

First, to obtain similarity data between different CSV files, use the code in the similarity folder. The Python table similarity module can be run as a standalone program to obtain the similarity between a set of tables.

### Prerequisites

The following Python libraries are required:

* numpy (https://pypi.org/project/numpy/)
* python-Levenshtein (https://pypi.org/project/python-Levenshtein/)
* scipy (https://pypi.org/project/scipy/)
* gensim (https://pypi.org/project/gensim/)
* transformers (https://pypi.org/project/transformers/)

The pre-trained models are available in the following links:

* fastText: https://fasttext.cc/docs/en/english-vectors.html
* Google word2vec: https://code.google.com/archive/p/word2vec/

The WikiTables files to build the task-specific model are available here: http://websail-fe.cs.northwestern.edu/TabEL/

### Installing and Deployment

First, check that pip package-management system is present in your system. Otherwise install it following this instructions: https://pip.pypa.io/en/stable/installing/

Download the table_similarity module and run the following commands:

* To install the libraries required: $ pip install -r requirements.txt
* To install the table_similarity module: $ python setup.py install

### Configuring and running the similarity module

The module reads the parameters from a configuration file ("config.ini"). This file contains four sections to define the model and alpha values ("Setup"), the directories where the models and input tables are stored ("Directories"), the output file with the similarity values ("Files") and the filenames of the models ("Models"). This is an example of "config.ini":

```
[Setup] 
model_names = fasttext
model_contents = fasttext
alpha = 0.5

[Directories]
models = ./models
tables = ./tables

[Files]
output = ./similarity.json

[Models]
wikitables_names = wikitables_names
wikitables_contents = wikitables_contents
google = GoogleNews-vectors-negative300.bin
fasttext = wiki-news-300d-1M.vec
bert = bert-base-uncased
roberta = roberta-base
```

This example uses *fastText* for both column names ("model_names") and content ("model_content"). The parameter alpha is set to "0.5" and the results are stored in a file named "similarity.json".

To run the program, import the table_similarity module in your code or run from the command line:

```
$ python table_similarity.py
```

## Authors

* **David Tomás** 
* **José Pilaluisa** 
* **Borja Navarro-Colorado** 
* **Jose-Norberto Mazón** 

