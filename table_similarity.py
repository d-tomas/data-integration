from collections import defaultdict
import configparser
import csv
from gensim.models import KeyedVectors
import json
import Levenshtein as lev
import numpy as np
from operator import itemgetter
import os
import re
from scipy import spatial
import string
import sys
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel


def normalize(text):
    """
    Take a string and split camelCased-words, remove punctuation, extra whitespaces, lowercase, etc.

    :param text: string to normalize
    :return: text normalized
    """
    split_words = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text)).split()
    text = ' '.join(split_words)  # Split camelCased-words
    text = text.translate(str.maketrans('_-/', '   '))  # Split words including '-', '_' and '/'
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra whitespaces
    text = text.lower()  # Lowercase

    return text


def load_tables(tables_dir):
    """
    For every CSV file in a directory, create a dictionary with table names as keys, which contain a dictionary with column headings as keys and cell values as content

    :param tables_dir: directory storing the tables in CSV format, with header, delimiter ',' and quotechar '"'
    :return: a dictionary with all the content of the tables
    """
    dict_tables = defaultdict()
    for file_name in os.listdir(tables_dir):
        with open(os.path.join(tables_dir, file_name)) as input_file:
            csv_reader = csv.DictReader(input_file, delimiter = ',', quotechar = '"')
            dict_tables[file_name] = defaultdict(list)
            for row in csv_reader:
                for column_name in row:
                    dict_tables[file_name][normalize(column_name)].append(normalize(row[column_name]))

    return dict_tables


def get_mean_vector(model_contents, cell_values):
    """
    Given the cell values for a column, compute the mean vector of the word embeddings of each cell value
    Uses non-contextual word embeddings (Word2vec and fastText)

    :param model_contents: word embedding model to obtain the vector for each cell value
    :param cell_values: list of cell values of a column
    :return: an embedding vector representing the mean of all the cell values in the column
    """
    list_tokens = []
    for token in cell_values:
        list_tokens.extend(token.split())  # Extract tokens from attributes with multiple words

    cell_values = [normalize(x) for x in list_tokens if normalize(x) in model_contents]  # Remove out-of-vocabulary words after normalization
    if len(cell_values) >= 1:
        return np.mean(model_contents[cell_values], axis=0)
    else:
        return []


def get_mean_vector_bert(model_contents, cell_values, tokenizer):
    """
    Given the cell values for a column, compute a word embedding vector representing these values
    Uses contextual word embeddings (BERT, RoBERTa and WikiTables)

    :param model_contents: BERT-like model to obtain a single vector for all the cell values
    :param cell_values: list of values of a column
    :param tokenizer: tokenizer function for BERT-like model
    :return: an embedding vector representing all the cell values for a column
    """
    list_tokens = []
    cell_values = list(set(cell_values))  # Remove duplicates
    for token in cell_values:
        list_tokens.extend(token.split())  # Extract tokens from cells with multiple words

    cell_values = [normalize(x) for x in list_tokens]  # There are no OOV words in contextual word embeddings
    if len(cell_values) >= 1:
        list_ids = tokenizer.encode(' '.join(cell_values), return_tensors='pt')

        # The number of elements must be cut to a maximum of 510 (+2 for special tokens, since BERT-like models input limit is 512)
        if list(list_ids[0].size())[0] > 512:
            list_ids_2 = list_ids[:, :512]
            list_ids_2[0][-1] = list_ids[0][-1]  # Include the '[SEP]' token at the end
            list_ids = list_ids_2

        return model_contents(list_ids)[0][0][0].tolist()  # The first token ('[CLS]') encodes the sentence embedding
    else:
        return []


def create_vectors_content(model_contents, dict_tables):
    """
    Creates a dictionary where every column is represented by an embedding vector based on cell values
    Uses non-contextual word embeddings (Word2vec and fastText)

    :param model_contents: word embedding model to calculate the vector of the cell values
    :param dict_tables: dictionary that contains all the cell values of the columns for each table
    :return: a dictionary that stores, for every table in the directory, an embedding vector for every column
    """
    vectors_content = defaultdict()
    for table in dict_tables:
        vectors_content[table] = defaultdict()
        for column in dict_tables[table]:
            vectors_content[table][column] = get_mean_vector(model_contents, dict_tables[table][column])

    return vectors_content


def create_vectors_content_bert(model_contents, dict_tables, tokenizer):
    """
    Creates a dictionary where every column is represented by an embedding vector based on cell values
    Uses contextual word embeddings (BERT, RoBERTa and WikiTables)

    :param model_contents: BERT-like model to calculate the vector of the column values
    :param dict_tables: dictionary that contains all the cell values for each table
    :param tokenizer: tokenizer function for BERT-like model
    :return: a dictionary that stores, for every table in the directory, a word embedding vector for each column
    """
    vectors_content = defaultdict()
    for table in dict_tables:
        vectors_content[table] = defaultdict()
        for column in dict_tables[table]:
            vectors_content[table][column] = get_mean_vector_bert(model_contents, dict_tables[table][column], tokenizer)

    return vectors_content


def create_vectors_names_bert(model_names, dict_tables, tokenizer):
    """
    Creates a dictionary where every column is represented by an embedding vector based on column headings
    Uses contextual word embeddings (BERT, RoBERTa and WikiTables)

    :param model_names: BERT-like model to calculate the vector of column headings
    :param dict_tables: dictionary that contains all columns and their cell values for each table
    :param tokenizer: tokenizer function for BERT-like model
    :return: a dictionary that stores, for every table in the directory, a word embedding vector for each column
    """
    vectors_names = defaultdict()
    for table in dict_tables:
        vectors_names[table] = defaultdict()
        tokens = []
        for column in dict_tables[table]:
            tokens.extend(column.split())
        # Get the word embedding representation for each column
        tensor_table_ids = tokenizer.encode(' '.join(tokens), return_tensors='pt')
        # Limit to 510 tokens (maximum token length for BERT-like models is 512, including special tokens '[CLS]' and '[SEP]')
        if list(tensor_table_ids[0].size())[0] > 512:
            tensor_table_ids_2 = tensor_table_ids[:, :512]
            tensor_table_ids_2[0][-1] = tensor_table_ids[0][-1]  # Include the '[SEP]' token at the end
            tensor_table_ids = tensor_table_ids_2

        list_table_ids = tensor_table_ids[0].tolist()
        features = model_names(tensor_table_ids)[0][0]

        # Get tokens for each column
        for column in dict_tables[table]:
            tensor_column_ids = tokenizer.encode(column, return_tensors='pt')
            list_column_ids = tensor_column_ids[0].tolist()

            # Get the word embedding for the column heading (average all the tokens in the column heading)
            list_embeddings = []
            for ids in list_column_ids[1:-1]:  # Discard the first token ('[CLS]') and the last one ('[SEP]')
                list_embeddings.append(features[list_table_ids.index(ids)].detach().numpy())
            vectors_names[table][column] = np.mean(list_embeddings, axis=0)

    return vectors_names


def calculate_similarity(model_names, vectors_content, alpha, table_name_1, table_name_2, column_name_1, column_name_2):
    """
    Calculates similarity between two columns using non-contextual word embeddings (Word2vec and fastText)

    :param model_names: word embedding model to compare column headings
    :param vectors_content: word embedding vectors of the cell values
    :param alpha: specifies the weight in the final formula of the column headings and  cell values similarities
    :param table_name_1: name of the first table to compare
    :param table_name_2: name of the second table to compare
    :param column_name_1: name of the column from the first table to compare
    :param column_name_2: name of the column from the second table to compare
    :return: float value indicating the degree of similarity in the interval [0, 1]
    """
    # Keep the words from the column heading in the vocabulary of model_names (i.e. remove OOV tokens)
    tokens_1 = [token for token in column_name_1.split() if token in model_names]
    tokens_2 = [token for token in column_name_2.split() if token in model_names]

    # Similarity between column headings. If there is no coverage in model_names, calculate Levenshtein distance as a fallback
    if tokens_1 and tokens_2:
        similarity_names = model_names.n_similarity(tokens_1, tokens_2)  # Computes the average of the vectors and then the cosine similarity
    else:
        similarity_names = lev.ratio(column_name_1, column_name_2)  # Computes Levenshtein distance

    # Similarity between cell values. If there is no coverage, keep only the similarity of column headings
    if len(vectors_content[table_name_1][column_name_1]) == 300 and len(vectors_content[table_name_2][column_name_2]) == 300:
        # Calculate the cosine similarity between two word embedding vectors
        similarity_content = 1 - spatial.distance.cosine(vectors_content[table_name_1][column_name_1], vectors_content[table_name_2][column_name_2])
        similarity = alpha * similarity_names + (1 - alpha) * similarity_content
    else:
        similarity = similarity_names

    return float(similarity)  # To avoid problems with float32 values that are not JSON serializable


def calculate_similarity_bert(vectors_names, vectors_content, alpha, table_name_1, table_name_2, column_name_1, column_name_2, tokenizer):
    """
    Calculates the similarity between two columns using contextual word embeddings (BERT, RoBERTa and WikiTables)

    :param vectors_names: word embedding vectors of the column headings
    :param vectors_content: word embedding vectors of the cell values
    :param alpha: specifies the weight in the final formula of the column headings and  cell values similarities
    :param table_name_1: name of the first table to compare
    :param table_name_2: name of the second table to compare
    :param column_name_1: name of the column from the first table to compare
    :param column_name_2: name of the column from the second table to compare
    :param tokenizer: tokenizer function for the contextual model
    :return: float value indicating the degree of similarity in the interval [0, 1]
    """
    # Similarity between column headings computing the cosine similarity on contextual word embeddings
    similarity_names = 1 - spatial.distance.cosine(vectors_names[table_name_1][column_name_1], vectors_names[table_name_2][column_name_2])

    # Similarity between cell values. If there is no coverage, keep only the similarity of column headings
    if len(vectors_content[table_name_1][column_name_1]) > 0 and len(vectors_content[table_name_2][column_name_2]) > 0:
        # Calculate the cosine similarity between two word embeddings vectors
        similarity_content = 1 - spatial.distance.cosine(vectors_content[table_name_1][column_name_1], vectors_content[table_name_2][column_name_2])
        similarity = alpha * similarity_names + (1 - alpha) * similarity_content
    else:
        similarity = similarity_names

    return float(similarity)  # To avoid problems with float32 values that are not JSON serializable


def load_models(model_names_cfg, model_contents_cfg, dict_models_files):
    """
    Loads in memory the word embeddings models passed as parameter (one for column headings, the other for cell values)

    :param model_names_cfg: name of the model for column headings
    :param model_contents_cfg: name of the model for the cell values
    :param dict_models_files: path to the word embedding models
    :return: models loaded in memory
    """
    # Load model to encode column headings
    if model_names_cfg == 'wikitables-names':  # BERT-base uncased model fine-tuned on WikiTables corpus
        model_names = AutoModel.from_pretrained(dict_models_files[model_names_cfg])
    elif model_names_cfg == 'google':  # Word2vec model
        model_names = KeyedVectors.load_word2vec_format(dict_models_files[model_names_cfg], binary=True)
    elif model_names_cfg == 'fasttext':  # fastText model
        model_names = KeyedVectors.load_word2vec_format(dict_models_files[model_names_cfg])
    elif model_names_cfg == 'bert':  # BERT-base uncased
        model_names = BertModel.from_pretrained(dict_models_files[model_names_cfg])
    elif model_names_cfg == 'roberta':  # RoBERTa-base model
        model_names = RobertaModel.from_pretrained(dict_models_files[model_names_cfg])
    else:
        print('ERROR: wrong model name "' + model_names_cfg + '"')
        sys.exit(1)

    # Load model to encode cell values. If the model is the same in both cases, just assign variables
    if model_names_cfg == model_contents_cfg:
        model_contents = model_names
    elif model_contents_cfg == 'wikitables-contents':
        model_contents = AutoModel.from_pretrained(dict_models_files[model_contents_cfg])
    elif model_contents_cfg == 'google':
        model_contents = KeyedVectors.load_word2vec_format(dict_models_files[model_contents_cfg], binary=True)
    elif model_contents_cfg == 'fasttext':
        model_contents = KeyedVectors.load_word2vec_format(dict_models_files[model_contents_cfg])
    elif model_contents_cfg == 'bert':
        model_contents = BertModel.from_pretrained(dict_models_files[model_contents_cfg])
    elif model_contents_cfg == 'roberta':
        model_contents = RobertaModel.from_pretrained(dict_models_files[model_contents_cfg])
    else:
        print('ERROR: wrong model content "' + model_contents_cfg + '"')
        sys.exit(1)

    return model_names, model_contents


def main():
    # Load configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Alpha
    alpha = float(config['Setup']['alpha'])
    model_names_cfg = config['Setup']['model_names']
    model_contents_cfg = config['Setup']['model_contents']
    output_file = config['Files']['output']
    models_dir = config['Directories']['models']
    tables_dir = config['Directories']['tables']

    # Store paths to models
    dict_models_files = defaultdict()
    dict_models_files['wikitables-names'] = os.path.join(models_dir, config['Models']['wikitables-names'])
    dict_models_files['wikitables-contents'] = os.path.join(models_dir, config['Models']['wikitables-contents'])
    dict_models_files['google'] = os.path.join(models_dir, config['Models']['google'])
    dict_models_files['fasttext'] = os.path.join(models_dir, config['Models']['fasttext'])
    dict_models_files['bert'] = config['Models']['bert']
    dict_models_files['roberta'] = config['Models']['roberta']

    print('----------')
    print('Parameters')
    print('Model names: ' + model_names_cfg)
    print('Model contents: ' + model_contents_cfg)
    print('Alpha: ' + str(alpha))
    print('----------\n')

    print('Loading models...', end=' ')
    model_names, model_contents = load_models(model_names_cfg, model_contents_cfg, dict_models_files)
    print('Ok')

    print('Reading tables...', end=' ')
    dict_tables = load_tables(tables_dir)
    print('Ok')

    print('Creating word embeddings of cell values...', end=' ')
    if model_contents_cfg == 'bert':
        tokenizer = BertTokenizer.from_pretrained(dict_models_files[model_names_cfg])
        vectors_content = create_vectors_content_bert(model_contents, dict_tables, tokenizer)
    elif model_contents_cfg == 'roberta':
        # add_prefix_space avoids words to have different representation if they are at the beginning of the text or not
        tokenizer = RobertaTokenizer.from_pretrained(dict_models_files[model_contents_cfg], add_prefix_space=True)
        vectors_content = create_vectors_content_bert(model_contents, dict_tables, tokenizer)
    elif model_contents_cfg == 'wikitables-contents' or model_contents_cfg == 'wikitables-names':
        tokenizer = AutoTokenizer.from_pretrained(dict_models_files[model_contents_cfg])
        vectors_content = create_vectors_content_bert(model_contents, dict_tables, tokenizer)
    else:  # Non-contextual models
        vectors_content = create_vectors_content(model_contents, dict_tables)
    print('Ok')

    # For contextual word embeddings, also obtain the word embeddings of column headings
    if model_names_cfg == 'bert' or model_names_cfg == 'roberta' or model_names_cfg == 'wikitables-contents' or model_names_cfg == 'wikitables-names':
        print('Creating word embedding of column headings...', end=' ')
        vectors_names = create_vectors_names_bert(model_names, dict_tables, tokenizer)
        print('Ok')

    dict_results = {}
    counter = 1
    for table_name_1 in dict_tables:
        print("Processing table " + str(counter))
        counter+=1
        table_similarity = defaultdict()
        for table_name_2 in dict_tables:
            if table_name_1 != table_name_2 and (table_name_2 not in dict_results or table_name_1 not in dict_results[table_name_2]):  # Avoid comparing a table with itself
                table_similarity[table_name_2] = []
                for column_name_1 in dict_tables[table_name_1]:
                    for column_name_2 in dict_tables[table_name_2]:
                        if model_names_cfg == 'bert' or model_names_cfg == 'roberta' or model_names_cfg == 'wikitables-names' or model_names_cfg == 'wikitables-contents':
                            similarity = calculate_similarity_bert(vectors_names, vectors_content, alpha, table_name_1, table_name_2, column_name_1, column_name_2, tokenizer)
                        else:
                            similarity = calculate_similarity(model_names, vectors_content, alpha, table_name_1, table_name_2, column_name_1, column_name_2)
                        table_similarity[table_name_2].append((column_name_1, column_name_2, similarity))
                    table_similarity[table_name_2] = sorted(table_similarity[table_name_2], key=itemgetter(2), reverse=True)
        if table_similarity:
            dict_results[table_name_1] = table_similarity

    print('Saving to file "' + output_file + '"...', end=' ')
    with open(output_file, 'w') as file:
        json.dump(dict_results, file)
    print('Ok')


if __name__ == '__main__':
    main()
