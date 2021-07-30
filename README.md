# Contextual word embeddings for tabular data search and integration

This code allows computing table similarity based on differnt contextual and non-contextual word embedding models. This similarity can be further used for *ad hoc* table retrieval and integration (*join* and *union* operations).

## Getting Started

The retrieval and integration pipeline is divided in three processes. Each one has a specific folder:

1. [Calculate the similarity between tables](https://github.com/d-tomas/data-integration/tree/main/similarity)
2. [Retrieve the most similar tables for a given table](https://github.com/d-tomas/data-integration/tree/main/retrieval)
3. [Integrate using union and join operations](https://github.com/d-tomas/data-integration/tree/main/integration)

## Prerequisites

The following Python libraries are required:

* [numpy](https://pypi.org/project/numpy/)
* [python-Levenshtein](https://pypi.org/project/python-Levenshtein/)
* [scipy](https://pypi.org/project/scipy/)
* [gensim](https://pypi.org/project/gensim/)
* [transformers](https://pypi.org/project/transformers/)

The pre-trained models are available in the following links:

* [fastText](https://fasttext.cc/docs/en/english-vectors.html)
* [Google word2vec](https://code.google.com/archive/p/word2vec/)

The WikiTables files to build the task-specific model are available [here](http://websail-fe.cs.northwestern.edu/TabEL/).

## Table similarity

To calculate the similarity between different CSV files, use the code in the [similarity](https://github.com/d-tomas/data-integration/tree/main/similarity) folder. The Python table similarity module can be run as a standalone program to obtain the similarity between a set of tables.

### Installing and Deployment

First, check that `pip` package-management system is present in your system. Otherwise install it following this instructions: https://pip.pypa.io/en/stable/installing/

Download the table_similarity module and run the following commands:

* To install the libraries required: `$ pip install -r requirements.txt`
* To install the `table_similarity` module: `$ python setup_similarity.py install`

### Configuring and running the similarity module

The module reads the parameters from a configuration file (`config.ini`). This file contains four sections to define the model and alpha values (`Setup`), the directories where the models and input tables are stored (`Directories`), the output file with the similarity values (`Files`) and the filenames of the models (`Models`). This is an example of `config.ini`:

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

This example uses *fastText* for both column names (`model_names`) and content (`model_content`). The parameter alpha is set to `0.5` and the results are stored in a file named `similarity.json`.

To run the program, import the `table_similarity` module in your code or run from the command line:

```
$ python table_similarity.py
```

## Table retrieval

Once the similarity is calculated, this module allows retrieving the most relevant tables for a given table. The code is in the [retrieval](https://github.com/d-tomas/data-integration/tree/main/similarity) folder. This module can be run as a standalone program, but requieres as an input the similarity file generated by the `table_similarity` module.

### Installing and Deployment

Download the `table_retrieval` module and run the following command:

```
$ python setup_retrieval.py install
```

### Configuring and running the similarity module

Again, the module reads the parameters from a configuration file (`config.ini`). This file contains three sections to define the task at hand (`union_retrieval` or `join_retrieval`) and the number of documents to be retrieved (`Setup`), the directory where the input tables are stored (`Directories`), and the file with the similarity between tables and the output file generated by this module (`Files`). This is an example of `config.ini`:

```
[Setup]
task = union_retrieval
n = 10

[Directories]
tables = ./tables

[Files]
tables_similarity = ./similarity.json
output = ./retrieval.json
```

This example performs union retrieval, i.e., retrieves the most suitable tables for further integration using *union* operation. For each table, the program will return the top 10 ranked tables. The results are stored in a file named `retrieval.json`.

To run the program, import the `table_retrieval` module in your code or run from the command line:

```
$ python table_retrieval.py
```

## Table integration

Giving two tables, this module identifies which pairs of columns can be matched in *union* and *join* operations for tabular data integration. The code is in the [integration](https://github.com/d-tomas/data-integration/tree/main/integration) folder. This module can be run as a standalone program, but requieres as an input the similarity file generated by the `table_similarity` module.

### Installing and Deployment

Download the `table_integration` module and run the following command:

```
$ python setup_integration.py install
```

### Configuring and running the similarity module

Again, the module reads the parameters from a configuration file (`config.ini`). This file contains three sections to define the operation (`union` or `join`) to be done (`Setup`), the directory where the input tables are stored (`Directories`), and the file with the similarity between tables and the output file generated by this module (`Files`). This is an example of `config.ini`:

```
[Setup]
task = union

[Directories]
tables = ./tables

[Files]
tables_similarity = ./similarity.json
output = ./integration.json
```

This example performs the union operation. For each pair of tables, the program will return the columns that will be matched in a *union* operation. The results are stored in the file `integration.json`.

To run the program, import the `table_integration` module in your code or run from the command line:

```
$ python table_integration.py
```


## Authors

* David Tomás
* José Pilaluisa
* Borja Navarro-Colorado
* Jose-Norberto Mazón

