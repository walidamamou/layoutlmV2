import pandas as pd
import numpy as np
import os
import argparse

from PIL import Image
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D,Dataset
from transformers import LayoutLMv2Processor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return (f.readlines())

def get_zip_dir_name():
  os.chdir('/content/data')
  dir_list = os.listdir()
  any_file_name = dir_list[0]
  zip_dir_name = any_file_name[:any_file_name.find('\\')]
  if all(list(map(lambda x : x.startswith(zip_dir_name),dir_list))):
    return zip_dir_name
  return False

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  
  encoded_inputs = processor(images,words, boxes=boxes, word_labels=word_labels,
                             padding="max_length", truncation=True)
  
  return encoded_inputs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    TEST_SIZE = float(args.valid_size)
    OUTPUT_PATH = args.output_path

    os.makedirs(args.output_path,exist_ok=True)
    os.chdir('/content/data')
    files = {}
    zip_dir_name = get_zip_dir_name()
    if zip_dir_name:
        files['train_box'] = read_text_file(os.path.join(os.curdir,f'{zip_dir_name}\\{zip_dir_name}_box.txt'))
        files['train_image'] = read_text_file(os.path.join(os.curdir,f'{zip_dir_name}\\{zip_dir_name}_image.txt'))
        files['train'] = read_text_file(os.path.join(os.curdir,f'{zip_dir_name}\\{zip_dir_name}.txt'))
    else:
        for f in os.listdir():
            if f.endswith('.txt') and f.find('box') != -1:
                files['train_box'] = read_text_file(os.path.join(os.curdir,f))
            elif f.endswith('.txt') and f.find('image') != -1:
                files['train_image'] = read_text_file(os.path.join(os.curdir,f))
            elif f.endswith('.txt') and f.find('labels') == -1:
                files['train'] = read_text_file(os.path.join(os.curdir,f))

    assert(len(files['train']) == len(files['train_box']))
    assert(len(files['train_box']) == len(files['train_image']))
    assert(len(files['train_image']) == len(files['train']))

    images = {}
    for i,row in enumerate(files['train_image']):
        if row != '\n':
            image_name = row.split('\t')[-1]
            images.setdefault(image_name.replace('\n',''),[]).append(i)

    words,bboxes,ner_tags,image_path= [],[],[],[]
    for image,rows in images.items():
        words.append([row.split('\t')[0].replace('\n','') for row in files['train'][rows[0]:rows[-1]+1]])
        ner_tags.append([row.split('\t')[1].replace('\n','') for row in files['train'][rows[0]:rows[-1]+1]])
        bboxes.append([box.split('\t')[1].replace('\n','') for box in files['train_box'][rows[0]:rows[-1]+1]])
        if zip_dir_name:
            image_path.append(f"/content/data/{zip_dir_name}\\{image}")
        else:
            image_path.append(f"/content/data/{image}")

    labels = list(set([tag for doc_tag in ner_tags for tag in doc_tag]))  
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    class data_config:
        labels = np.unique([tag for doc_tag in ner_tags for tag in doc_tag]).tolist()
        num_labels = len(labels)
        id2label = {v: k for v, k in enumerate(labels)}
        label2id = {k: v for v, k in enumerate(labels)}

    dataset_dict = {
        'id' : range(len(words)),
        'words' : words,
        'bboxes' : [[list(map(int,bbox.split())) for bbox in doc]  for doc in bboxes],
        'ner_tags' : [[data_config.label2id[tag] for tag in ner_tag] for ner_tag in ner_tags],
        'image_path' : image_path
    }


    df = pd.DataFrame(dataset_dict)
    df.id = df.id.apply(str)
    assert df.shape[0] == len(images)
    train, valid = train_test_split(df,test_size=TEST_SIZE,shuffle=True)

    #filter out untagged images to prevent 'O' tag overfitting (UNDERSAMPLINIG)
    limit = 25 # we do not accept to reduce the train dataset with more than 25% of it's original size 
    untagged_images = pd.DataFrame(columns=train.columns)
    filtered_train = train.copy()
    for index,row in filtered_train.iterrows():
        tags = row['ner_tags']
        if all([tag == data_config.label2id['O'] for tag in tags]):
            untagged_images = untagged_images.append(row)
            filtered_train = filtered_train.drop(index = index)
            if filtered_train.shape[0] <= ((100-limit)/100) * train.shape[0]:
                break

    features = Features({
        'id': Value(dtype='string', id=None),
        'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'bboxes' : Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
        'ner_tags' : Sequence(feature=ClassLabel(num_classes=len(data_config.labels),names=data_config.labels, names_file=None, id=None), length=-1, id=None),
        'image_path' : Value(dtype='string', id=None)
    })

    train_dataset = Dataset.from_pandas(filtered_train,features=features)
    valid_dataset = Dataset.from_pandas(valid,features=features)

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    # we need to define custom features
    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(ClassLabel(names=data_config.labels)),
    })


    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names,
                                        features=features)
    valid_dataset = valid_dataset.map(preprocess_data, batched=True, remove_columns=valid_dataset.column_names,
                                        features=features)
                                        
    train_dataset.set_format(type="torch", device="cuda")
    valid_dataset.set_format(type="torch", device="cuda")

    if not OUTPUT_PATH.endswith('/'):
        OUTPUT_PATH+='/'
    train_dataset.save_to_disk(f'{OUTPUT_PATH}train_split')
    valid_dataset.save_to_disk(f'{OUTPUT_PATH}eval_split')
