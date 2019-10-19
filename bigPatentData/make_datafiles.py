import sys
import os
import hashlib
import subprocess
import collections
import argparse
from os.path import join

import json
import tarfile
import io
import pickle as pkl
import nltk
from nltk.tokenize import sent_tokenize

from readData import SPLITS


dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]


def read_patent_file(path, index, part):
    if part == 'features':
        extension = '.desc'
    elif part == 'labels':
        extension = '.label'
    else:
        raise ValueError('wrong part')
    with open(join(path, part, f'{index}'+extension), "r") as f:
        lines = sent_tokenize(f.read())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_sents(path, index, part):
    """ return as list of sentences"""
    lines = read_patent_file(path, index, part)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.lower().strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        else:
            article_lines.append(line)

    return article_lines


def get_count(path):
    f_len = len(os.listdir(join(path, 'features')))
    l_len = len(os.listdir(join(path, 'labels')))
    assert f_len == l_len
    return f_len


def write(input_path, split):
    """Reads the tokenized patent files and writes them to an out_file.
    """
    print("Writing {}\n".format(split))
    path = join(input_path, 'tokens', split)
    output_path = join(input_path, 'finished_files', split)
    os.makedirs(output_path, exist_ok=True)
    count = get_count(path)

    if split == 'train':
        vocab_counter = collections.Counter()

    for i in range(count):
        if i % 1000 == 0 or i == count-1:
            print("Writing file {} of {}\n".format(i+1, count))
        # Get the strings to write to .bin file
        article_sents = get_sents(path, i, 'features')
        abstract_sents = get_sents(path, i, 'labels')

        # Write to JSON file
        js_example = {}
        js_example['id'] = i
        js_example['article'] = article_sents
        js_example['abstract'] = abstract_sents
        js_serialized = json.dumps(js_example, indent=4).encode()
        with open(join(output_path, f'{i}.json'), 'wb') as file:
            file.write(js_serialized)

        # Write the vocab to file, if applicable
        if split == 'train':
            art_tokens = ' '.join(article_sents).split()
            abs_tokens = ' '.join(abstract_sents).split()
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens] # strip
            tokens = [t for t in tokens if t != ""] # remove empty
            vocab_counter.update(tokens)

    print("Finished writing {}\n".format(split))
    # write vocab to file
    if split == 'train':
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data')
    parser.add_argument('--input_path', type=str, help='path to data')
    args = parser.parse_args()
    for split in SPLITS:
        write(args.input_path, split)
