import json
import gzip
import os
import sys
import argparse


CPC_CODES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'y']
SPLITS = ['train', 'val', 'test']


def format_data(input_path):
    for split in SPLITS:
        i = 0
        for code in CPC_CODES:
            for patent in readData(input_path, split, code):
                save_patent(input_path, split, patent, i)
                i += 1


def save_patent(path, split, patent, index):
    output_path = os.path.join(path, 'finished_files', split)
    os.makedirs(output_path, exist_ok=True)
    patent['article'] = patent.pop('description')
    with open(os.path.join(output_path, f'{index}.json'), 'w') as file:
        file.write(json.dumps(patent))


def patent_count(path):
    input_path, split = path.rsplit('/', 1)
    count = 0
    for code in CPC_CODES:
        count += len(list(readData(input_path, split, code)))
    return count


def readData(input_path,split_type,cpc_code):
    file_names = os.listdir(os.path.join(input_path,split_type,cpc_code))
    # reading one of the gz files.
    for file_name in file_names:
        print("Reading file "+ file_name + " from "+ split_type+" split for cpc code " + cpc_code)
        with gzip.open(os.path.join(input_path,split_type,cpc_code,file_name),'r') as fin:
            for row in fin:
                content = row.decode('utf-8')
                json_obj = json.loads(content)
                yield json_obj


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read data')
    parser.add_argument('--cpc_code', type=str, help='can be a, b,c,d,e,f,g,h,y')
    parser.add_argument('--split_type', type=str, help='can be train, test, val')
    parser.add_argument('--input_path', type=str, help='path to data')

    args = parser.parse_args()
    split_type = args.split_type
    cpc_code = args.cpc_code
    input_path = args.input_path

    format_data(input_path)
