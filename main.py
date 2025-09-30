import os
import re
from collections import Counter

import pandas as pd
import numpy as np
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BOOK_FILES = {
    "the_adventures_of_huckleberry_finn": "books/the_adventures_of_huckleberry_finn.txt",
    "the_great_gatsby": "books/the_great_gatsby.txt",
    "war_and_peace": "books/war_and_peace.txt",
    "crime_and_punishment": "books/crime_and_punishment.txt"
}

def clean_book(filename):
    with open(filename, encoding='utf-8', errors='ignore') as file:
        text = file.read()

    start_markers = [
        r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'START OF THE PROJECT GUTENBERG EBOOK'
    ]

    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():]
            break

    end_markers = [
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'END OF THE PROJECT GUTENBERG EBOOK'
    ]

    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
            break

    chapter_markers = [
        r'CHAPTER I\b',
        r'CHAPTER 1\b',
        r'Chapter I\b',
        r'Chapter 1\b',
        r'Chapter One\b',
        r'CHAPTER ONE\b'
    ]

    first_chapter_loc = len(text)
    for marker in chapter_markers:
        match = re.search(marker, text, re.MULTILINE)
        if match and match.start() < first_chapter_loc:
            first_chapter_loc = match.start()

    if first_chapter_loc < len(text):
        text = text[first_chapter_loc:]

    return text.strip()

def prep_doc(doc_text, doc_name='Book'):
    return [(doc_name, doc_text)]

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]+", "", text)
    text = re.sub(r'\d+', '', text)

    tokens = text.split()
    filtered_tokens = []

    lemmatizer = WordNetLemmatizer()

    for word in tokens:
        if word not in stopwords.words('english'):
            lemmatized_word = lemmatizer.lemmatize(word)
            filtered_tokens.append(lemmatized_word)

    return filtered_tokens

def create_table(docs):
    # docs: list of tuples(name, text) of cleaned books
    doc_tokens = {}
    all_words = set()
    for name, text in docs:
        tokens = tokenize(text)
        doc_tokens[name] = tokens
        all_words.update(tokens)

    word_counts = {}
    for name, text in doc_tokens.items():
        counter = Counter(text)
        word_counts[name] = counter

    df = pd.DataFrame(0, index=sorted(all_words), columns=list(doc_tokens.keys()))

    for name, counter in word_counts.items():
        for word, count in counter.items():
            df.loc[word, name] = count

    return df

def process_book(filename):
    doc_name = os.path.basename(filename).replace('.txt', '')

    text = clean_book(filename)
    docs = prep_doc(text, doc_name=doc_name)
    df = create_table(docs)

    return df


# Example usage
if __name__ == "__main__":
    for book_name, file_path in BOOK_FILES.items():
        df = process_book(file_path)

        top_10 = df[book_name].sort_values(ascending=False).head(10)
        print(top_10)

        output_csv = f'word_matrices/{book_name}.csv'
        df.to_csv(output_csv)


