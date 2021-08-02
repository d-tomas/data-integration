from collections import defaultdict
import configparser
import csv
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


def calculate_union(dict_similarity, threshold):
    """
    Identify the columns between two tables that match in a union operation

    :param dict_similarity: dictionary where keys are table names and content is a list of column pairs and their similarity
    :param threshold: cut-off similarity threshold to decide if union can be applied to a pair of columns
    :return: dictionary with the columns that can be part of a union for each pair of tables
    """
    dict_union = {}
    for table_1 in dict_similarity:
        dict_union[table_1] = {}
        for table_2 in dict_similarity[table_1]:
            dict_union[table_1][table_2] = []
            for column_pair in dict_similarity[table_1][table_2]:
                if column_pair[2] >= threshold:
                    if not contained(dict_union[table_1][table_2], column_pair):
                        dict_union[table_1][table_2].append(column_pair)
            if not dict_union[table_1][table_2]:
                del dict_union[table_1][table_2]

    return dict_union


def calculate_join(dict_similarity, threshold):
    """
    Identify tables that can be joined (there is at least one pair of columns with a similarity above the threshold)

    :param dict_similarity: dictionary where keys are table names and content is a list of column pairs and their similarity
    :param threshold: cut-off similarity threshold to decide if two tables can be joined
    :return: dictionary with the pair of columns used to join each pair of tables (where possible)
    """
    dict_join = {}
    for table_1 in dict_similarity:
        for table_2 in dict_similarity[table_1]:
            # Check only the pair of columns with the maximum similarity
            if dict_similarity[table_1][table_2][0][2] >= threshold:
                dict_join[table_1] = {}
                dict_join[table_1][table_2] = dict_similarity[table_1][table_2][0]
                continue
            else:
                break

    return dict_join


def main():
    # Load configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')

    task = config['Setup']['task'] # Integration task. Valid values are: "union" and "join"
    threshold = float(config['Setup']['threshold'])
    tables_similarity = config['Files']['tables_similarity']  # File with tables similarity
    output_file = config['Files']['output']  # Output file to store the tables retrieved in JSON format

    print('Loading similarity values...', end=' ')
    dict_similarity = load_similarity(tables_similarity)  # dict_similarity is already sorted from higher to lower similarity
    print('Ok')

    # Union operation
    if task == 'union':
        dict_union = calculate_union(dict_similarity, threshold)
        print('Saving to file "' + output_file + '"...', end=' ')
        with open(output_file, 'w') as file:
            json.dump(dict_union, file)
        print('Ok')
    # Join operation
    elif task == 'join':
        dict_join = calculate_join(dict_similarity, threshold)
        print('Saving to file "' + output_file + '"...', end=' ')
        with open(output_file, 'w') as file:
            json.dump(dict_join, file)
        print('Ok')


if __name__ == '__main__':
    main()
    