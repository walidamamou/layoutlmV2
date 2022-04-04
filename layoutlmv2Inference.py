import pandas as pd
import numpy as np
import torch
import os
import sys
import json
import logging

from PIL import Image, ImageDraw, ImageFont
from numpy.random import randint
from transformers import LayoutLMv2Processor

import warnings
import gc
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model= torch.load(sys.argv[1],map_location=device)
imag_path = sys.argv[2]

#Helper functions
def random_color():
  return np.random.randint(0,255,3)
def normalize_box(bbox,width,height):
  return [
          int(bbox[0]*(1000/width)),
          int(bbox[1]*(1000/height)),
          int(bbox[2]*(1000/width)),
          int(bbox[3]*(1000/height)),
  ]

def compare_boxes(b1,b2):
  b1 = np.array([c for c in b1])
  b2 = np.array([c for c in b2])
  equal = np.array_equal(b1,b2)
  return equal

def mergable(w1,w2):
  if w1['label'] == w2['label']:
    threshold = 7
    if abs(w1['box'][1] - w2['box'][1]) < threshold or abs(w1['box'][-1] - w2['box'][-1]) < threshold:
      return True
    return False
  return False

def main():
  os.system(f'tesseract "{imag_path}" /content/tsv_output -l eng tsv')
  inference_processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
  ocr_df = pd.read_csv("/content/tsv_output.tsv", sep='\t')
  ocr_df = ocr_df.dropna()
  ocr_df = ocr_df.drop(ocr_df[ocr_df.text.str.strip() == ''].index)
  text_output = ocr_df.text.tolist()
  doc_text = ' '.join(text_output)

  #read an image file for inference
  inference_image = Image.open(imag_path).convert('RGB')
  width, height = inference_image.size

  words = []
  for index,row in ocr_df.iterrows():
    word = {}
    origin_box = [row['left'],row['top'],row['left']+row['width'],row['top']+row['height']] 
    word['word_text'] = row['text']
    word['word_box'] = origin_box
    word['normalized_box'] = normalize_box(word['word_box'],width, height)
    words.append(word)

  boxlist = [word['normalized_box'] for word in words]
  wordlist = [word['word_text'] for word in words]

  encoding = inference_processor(inference_image,wordlist,boxes=boxlist,return_tensors="pt",padding="max_length", truncation=True) 
  for k,v in encoding.items():
    encoding[k] = v.to(device)

  model.eval()
  with torch.no_grad():
    inference_outputs = model(**encoding)
  inference_outputs.logits.shape

  raw_input_ids = encoding['input_ids'][0].tolist()
  predictions = inference_outputs.logits.argmax(-1).squeeze().tolist()
  token_boxes = encoding.bbox.squeeze().tolist()
  special_tokens = [inference_processor.tokenizer.cls_token_id, inference_processor.tokenizer.sep_token_id, inference_processor.tokenizer.pad_token_id]

  input_ids = [id for id in raw_input_ids if id not in special_tokens]
  predictions = [model.config.id2label[prediction] for i,prediction in enumerate(predictions) if not (raw_input_ids[i] in special_tokens)]
  actual_boxes = [box for i,box in enumerate(token_boxes) if not (raw_input_ids[i] in special_tokens )]

  assert(len(actual_boxes) == len(predictions))

  for word in words:
    word_labels = [] 
    token_labels = []
    word_tagging = None 
    for i,box in enumerate(actual_boxes,start=0):
      if compare_boxes(word['normalized_box'],box):
        if predictions[i] != 'O':
          word_labels.append(predictions[i][2:])
        else:
          word_labels.append('O')
        token_labels.append(predictions[i])
    if word_labels != []:
      word_tagging =  word_labels[0] if word_labels[0] != 'O' else word_labels[-1]
    else:
      word_tagging = 'O'
    word['word_labels'] = token_labels
    word['word_tagging'] = word_tagging

  filtered_words = [{'id':i,'text':word['word_text'],
                    'label':word['word_tagging'],
                    'box':word['word_box'],
                    'words':[{'box':word['word_box'],'text':word['word_text']}]} for i,word in enumerate(words) if word['word_tagging'] != 'O']

  merged_taggings = []
  for i,curr_word in enumerate(filtered_words):
    skip = False
    neighbors = lambda word:[neighbor for neighbor in filtered_words if mergable(word,neighbor)]
    for items in merged_taggings:
      for item in items:
        if item in neighbors(curr_word):
          skip = True
          break
      if skip:
        break
    if skip:
      continue
    merged_taggings.append(neighbors(curr_word))

  merged_words = []
  for i,merged_tagging in enumerate(merged_taggings):
    if len(merged_tagging) > 1:
      new_word = {}
      merging_word = " ".join([word['text'] for word in merged_tagging])
      merging_box = [merged_tagging[0]['box'][0]-5,merged_tagging[0]['box'][1]-10,merged_tagging[-1]['box'][2]+5,merged_tagging[-1]['box'][3]+10]
      new_word['text'] = merging_word
      new_word['box'] = merging_box
      new_word['label'] = merged_tagging[0]['label']
      new_word['id'] = filtered_words[-1]['id']+i+1
      new_word['words'] = [{'box':word['box'],'text':word['text']} for word in merged_tagging]
      merged_words.append(new_word)

  filtered_words.extend(merged_words)
  predictions = [word['label'] for word in filtered_words]
  actual_boxes = [word['box'] for word in filtered_words]
  unique_taggings = set(predictions)

  label2color = {f'{label}':f'rgb({random_color()[0]},{random_color()[1]},{random_color()[2]})' for label in unique_taggings}

  inference_image = Image.open(imag_path).convert('RGB')
  draw = ImageDraw.Draw(inference_image)
  font = ImageFont.load_default()
  taggings = {}
  for prediction, box in zip(predictions, actual_boxes):
      # predicted_label = iob_to_label(prediction).lower()
      draw.rectangle(box, outline=label2color[prediction])
      draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)  

  doc_name = os.path.basename(imag_path)
  output_path = sys.argv[3]
  os.makedirs(output_path,exist_ok=True)
  inference_image.save(f"{output_path}/imageOutput.png")
  dictionary = {"document name":doc_name,"document": doc_text , "form": filtered_words}
  with open(f"{output_path}/jsonOutput.json","w",encoding='utf8') as outfile:
      json.dump(dictionary, outfile,ensure_ascii=False)

if __name__ == '__main__':
  try:
    main()
  except Exception as err :
    print(err)
    os.makedirs('log',exist_ok=True)
    logging.basicConfig(filename='log/error_output.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    logger.error(err)
