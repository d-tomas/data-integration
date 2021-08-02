import configparser
from collections import defaultdict
import csv
from operator import itemgetter
import json
import os
import re
import string


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


def load_similarity(tables_similarity):
    """
    Loads into memory the contents of the JSON file that contains the similarity values between every column from every table.

    :param tables_similarity: JSON file with the similarity values. Keys are table names and content is a list of column pairs and their similarity
    :return: dictionary with the information read from file
    """
    with open(tables_similarity) as json_file:
        dict_similarity = json.load(json_file)

    return dict_similarity


def load_tables(dir_tables):
    """
    Given a directory with a set of tables, return a dictionary where keys are the names of the tables and contents are the list of columns for each table.

    :param dir_tables: directory where the tables are stored
    :return: dictionary with the names of tables as keys and list of normalized columns as content
    """
    dict_tables = defaultdict()
    for file_name in os.listdir(dir_tables):
        with open(os.path.join(dir_tables, file_name)) as input_file:
            csv_reader = csv.reader(input_file, delimiter=',', quotechar='"')
            headings = next(csv_reader)
            dict_tables[file_name] = [normalize(x) for x in headings]  # Read headings

    return dict_tables


def contained(list_columns, column_similarity):
    """
    Check if any of the two columns of the similarity are in the list of columns.

    :param list_columns: list of columns in the form [(colname1, colname2, value), (colname1, colname2, value), ...]
    :param column_similarity: column similarity in the form (colname1, colname2, value)
    :return: True or False depending on whether the column similarity is included in the list or not
    """
    for columns in list_columns:
        if column_similarity[0] in columns or column_similarity[1] in columns:
            return True

    return False


def get_total_similarity_union(list_columns):
    """
    Calculate the similarity between two tables based on the similarity of their columns.
    The weighting schema favours unionable tables: obtains the average similarity for all the columns of a table.

    :param list_columns: list of similarity pairs of columns in the form [(colname1, colname2, value), (colname1, colname2, value), ...]
    :return: similarity between tables
    """
    total = 0
    for columns in list_columns:
        total += columns[2]

    return total/len(list_columns)


def get_total_similarity_join(list_columns):
    """
    Calculate the similarity between two tables based on the similarity of their columns.
    The weighting schema favours joinable tables: get the column with the highest similarity

    :param list_columns: list of similarity pairs of columns in the form [(colname1, colname2, value), (colname1, colname2, value), ...]
    :return: similarity between tables
    """
    total_similarity = list_columns[0][2]  # Similarity of the first pair of columns (already sorted from higher to lower similarity)

    return total_similarity


def main():
    global column_positives
    global column_negatives
    
    # Load configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')

    task = config['Setup']['task']  # Retrieval task. Valid values are: "union_retrieval" and "join_retrieval"
    n = int(config['Setup']['n'])  # Number of tables to retrieve for each query
    dir_tables = config['Directories']['tables']  # Directory with all the tables and the file with the matching columns information
    tables_similarity = config['Files']['tables_similarity']  # File with tables similarity
    output_file = config['Files']['output']  # Output file to store the tables retrieved in JSON format

    print('Loading tables...', end=' ')
    dict_tables = load_tables(dir_tables)
    print('Ok')

    print('Loading similarity values...', end=' ')
    dict_similarity = load_similarity(tables_similarity) # dict_similarity is already sorted from higher to lower similarity
    print('Ok')

    dict_results = {}  # Store the top "n" tables retrieved for each query table
    for table_1 in dict_tables:
        list_results = []
        for table_2 in dict_tables:
            if table_1 != table_2:
                list_columns = []  # Keep track of the columns already considered to keep only the maximum similarity of a column with another one
                if table_1 in dict_similarity and table_2 in dict_similarity[table_1]:
                    list_column_similarity = dict_similarity[table_1][table_2]
                else:
                    list_column_similarity = dict_similarity[table_2][table_1]
                for column_similarity in list_column_similarity:
                    if not contained(list_columns, column_similarity):
                        list_columns.append(column_similarity)
                if task == 'union_retrieval':
                    list_results.append((table_2, get_total_similarity_union(list_columns)))
                else:
                    list_results.append((table_2, get_total_similarity_join(list_columns)))
        dict_results[table_1] = sorted(list_results, key=itemgetter(1), reverse=True)[:n]  # Store the values from higher to lower and keep the top "n"

    print('Saving to file "' + output_file + '"...', end=' ')
    with open(output_file, 'w') as file:
        json.dump(dict_results, file)
    print('Ok')


if __name__ == '__main__':
    main()
