import argparse
import torch
import dgcn
import numpy as np
from tqdm import tqdm
import json

log = dgcn.utils.get_logger()

def main(args):
    dgcn.utils.set_seed(args.seed)

    # Load data
    log.debug("Loading data from '%s'." % args.data)
    data = dgcn.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = dgcn.Dataset(data["train"], args.batch_size)
    devset = dgcn.Dataset(data["dev"], args.batch_size)
    testset = dgcn.Dataset(data["test"], args.batch_size)

    model_file = "./save/model.pt"
    model = dgcn.DialogueGCN(args).to(args.device)

    coach = dgcn.Coach(trainset, devset, testset, model, None, args)
    ckpt = torch.load(model_file)
    coach.load_ckpt(ckpt)

    # Set model to evaluation mode
    model.eval()

    # Run prediction
    predictions = predict_model(model, testset, args)
    
    # Convert predictions to emotion labels
    predicted_labels = convert_predictions_to_labels(predictions)

    # save_labels_to_file(predicted_labels, "predicted_labels.txt")
   
    insert_emotion_annotations("data/test_data.json", predicted_labels)


def predict_model(model, testset, args):
    model.eval()
    preds = []

    with torch.no_grad():
        for idx in tqdm(range(len(testset)), desc="Predicting"):
            data = testset[idx]
            # Move data to the correct device (GPU/CPU)
            for k, v in data.items():
                data[k] = v.to(args.device)

            # Get model predictions
            y_hat = model(data)
            # Collect predictions
            preds.append(y_hat.detach().cpu())  # Move predictions to CPU

        # Concatenate predictions into a numpy array
        preds = torch.cat(preds, dim=-1).numpy()

    return preds


def convert_predictions_to_labels(predictions):
    # Define label_to_idx and create an inverse mapping idx_to_label
    label_to_idx = {
        'Happy': 0,
        'Surprise': 1,
        'Sad': 2,
        'Anger': 3,
        'Disgust': 4,
        'Fear': 5,
        'Neutral': 6,
        'unknown': -1
    }
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Convert each prediction index to its corresponding emotion label
    predicted_labels = [idx_to_label.get(int(pred), 'unknown') for pred in predictions]

    return predicted_labels

def save_labels_to_file(predicted_labels, filename="predicted_labels.txt"):
    # 将每个标签写入文件，每行一个标签
    with open(filename, 'w') as file:
        for label in predicted_labels:
            file.write(f"{label}\n")


def insert_emotion_annotations(json_file, predicted_labels, output_file="test_data.json"):
    # 加载 JSON 文件数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化标签计数器
    index = 0  

    # 遍历每个剧集
    for episode, scenes in data.items():  # 遍历顶层的每个剧集
        for scene in scenes.values():     # 遍历每个剧集中的场景
            for dialog_id, dialog_data in scene["Dialog"].items():
                if index < len(predicted_labels):
                    dialog_data["EmoAnnotation"] = predicted_labels[index]
                    index += 1
                else:
                    break  # 防止超出 predicted_labels 的范围

    # 保存修改后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Updated data saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test.py")
    
    # Data and model paths
    parser.add_argument("--data", type=str, required=True, help="Path to the data")
    parser.add_argument("--model_file", type=str, default="./save/model.pt", help="Path to the trained model")
    
    # Training parameters (for reference)
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:7", help="Device (cpu/gpu)")
    
    # Model parameters
    parser.add_argument("--wp", type=int, default=10, help="Past context window size")
    parser.add_argument("--wf", type=int, default=10, help="Future context window size")
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of GCN")
    parser.add_argument("--rnn", type=str, default="lstm", choices=["lstm", "gru"], help="RNN cell type")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="Dropout rate for RNN layers")
    parser.add_argument("--class_weight", action="store_true", help="Use class weights in nll loss.")

    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    log.debug(args)

    main(args)
