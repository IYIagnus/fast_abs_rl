import sys
import os
import hashlib
import subprocess
import collections
import argparse

import json
import tarfile
import io
import pickle as pkl

from os.path import join

from readData import readData, SPLITS, CPC_CODES


def split_data(input_path):
    for split in SPLITS:
        output_path = join(input_path, 'split_files', split)
        os.makedirs(join(output_path, 'features'), exist_ok=True)
        os.makedirs(join(output_path, 'labels'), exist_ok=True)
        i = 0
        for code in CPC_CODES:
            for patent in readData(input_path, split, code):
                abstract = patent['abstract'].encode().decode()
                desc = patent['description'].encode().decode()
                with open(join(output_path, 'features', f'{i}.desc'), 'w') as file:
                    file.write(desc)
                with open(join(output_path, 'labels', f'{i}.label'), 'w') as file:
                    file.write(abstract)
                i += 1


def tokenize(input_path):
    for split in SPLITS:
        for part in ['features', 'labels']:
            tokenize_patents(join(input_path, 'split_files', split, part),
                             join(input_path, 'tokens', split, part))


def tokenize_patents(patents_dir, tokenized_patents_dir):
    """Maps a whole directory of files to a tokenized version using
       Stanford CoreNLP Tokenizer
    """
    print("Preparing to tokenize {} to {}...".format(patents_dir,
                                             tokenized_patents_dir))

    os.makedirs(tokenized_patents_dir, exist_ok=True)

    patents = os.listdir(patents_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in patents:
            f.write(
                "{} \t {}\n".format(
                    os.path.join(patents_dir, s),
                    os.path.join(tokenized_patents_dir, s)
                )
            )
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', '-options', 'untokenizable=noneDelete', 'mapping.txt']
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(patents), patents_dir, tokenized_patents_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized patents directory contains the same number of
    # files as the original directory
    num_orig = len(os.listdir(patents_dir))
    num_tokenized = len(os.listdir(tokenized_patents_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized patents directory {} contains {} files, but it "
            "should contain the same number as {} (which has {} files). Was"
            " there an error during tokenization?".format(
                tokenized_patents_dir, num_tokenized, patents_dir, num_orig)
        )
    print("Successfully finished tokenizing {} to {}.\n".format(
        patents_dir, tokenized_patents_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read data')
    parser.add_argument('--input_path', type=str, help='path to data')
    args = parser.parse_args()

    split_data(args.input_path)
    tokenize(args.input_path)
