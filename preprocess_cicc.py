import argparse

from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pickle
import json

import dgcn

log = dgcn.utils.get_logger()

def text_to_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据移动到GPU
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 返回最后一层的平均词向量
    return embeddings.cpu().numpy()  # 将结果移回CPU并转换为NumPy数组

def split():
    dgcn.utils.set_seed(args.seed)
    train, dev, test = [], [], []

    filename = 'data/train_data.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for juji, juji_data in tqdm(data.items(), desc="train"):
        for dialog, dialog_data in juji_data.items():
            speakers, labels, sentence ,text_embedding= [], [], [], []
            for per_sentence, per_sentence_data in dialog_data['Dialog'].items():
                speakers.append(per_sentence_data['Speaker'])
                labels.append(per_sentence_data['EmoAnnotation'])
                sentence.append(per_sentence_data['Text'])
                # 使用bert模型，将文本转话为词向量
                sentence_embeddings = text_to_bert_embeddings(per_sentence_data['Text'])
                text_embedding.append(sentence_embeddings)
            train.append(dgcn.Sample(dialog, speakers, labels,
                                 text_embedding, sentence))

    filename = 'data/val_data.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for juji, juji_data in tqdm(data.items(), desc="val"):
        for dialog, dialog_data in juji_data.items():
            speakers, labels, sentence, text_embedding = [], [], [], []
            for per_sentence, per_sentence_data in dialog_data['Dialog'].items():
                speakers.append(per_sentence_data['Speaker'])
                labels.append(per_sentence_data['EmoAnnotation'])
                sentence.append(per_sentence_data['Text'])
                # 使用bert模型，将文本转话为词向量
                sentence_embeddings = text_to_bert_embeddings(per_sentence_data['Text'])
                text_embedding.append(sentence_embeddings)
            dev.append(dgcn.Sample(dialog, speakers, labels,
                                     text_embedding, sentence))

    filename = 'data/test_data.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for juji, juji_data in tqdm(data.items(), desc="test"):
        for dialog, dialog_data in juji_data.items():
            speakers, labels, sentence, text_embedding = [], [], [], []
            for per_sentence, per_sentence_data in dialog_data['Dialog'].items():
                speakers.append(per_sentence_data['Speaker'])
                labels.append(per_sentence_data['EmoAnnotation'])
                sentence.append(per_sentence_data['Text'])
                # 使用bert模型，将文本转话为词向量
                sentence_embeddings = text_to_bert_embeddings(per_sentence_data['Text'])
                text_embedding.append(sentence_embeddings)
            test.append(dgcn.Sample(dialog, speakers, labels,
                                   text_embedding, sentence))


    return train, dev, test


def main(args):
    train, dev, test = split()
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    dgcn.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess_ccic.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:2")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    tokenizer = BertTokenizer.from_pretrained('./data/bert-base-chinese/')
    model = BertModel.from_pretrained('./data/bert-base-chinese/')

    model.to(device)
    main(args)
