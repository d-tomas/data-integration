from setuptools import setup

setup(name = 'table_similarity',
    version = '0.1',
    description = 'Calculates the similarity between CSV tables.',
    author = 'David Tomás',
    author_email = 'dtomas@dlsi.ua.es',
    packages = ['table_similarity'],
    install_requires = ['numpy>=1.15.1', 'python-Levenshtein>=0.12.0', 'scipy>=1.1.0', 'gensim>=3.8.0', 'transformers>=4.9.1', 'torch>=1.7.1'])
