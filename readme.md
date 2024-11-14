## 比赛
2024 CICC 社会认知与决策大赛
http://www.crowdhmt.com/scd/submit

## 运行环境
```
python==3.9.1
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
torchstat==0.0.7
torchvision==0.13.1+cu113
```
## 依赖包
```
argparse
tqdm
numpy
json
pickle
transformers
logging
sklearn
torch_scatter
collections
torch_geometric
```
## 参数配置
```
device=cuda:7 
epochs=50 
drop_rate=0.4 
weight_decay=1e-8 
batch_size=5 
learning_rate=0.0003
```
## Quickstart
Download the pre-trained bert-base-chinese model from this [link](https://huggingface.co/google-bert/bert-base-chinese/tree/main) and save it in the `/data/` directory.

Download dataset from this [link](https://pan.baidu.com/s/1KhaexKzVHKb9calgo8_BLA?pwd=ma6q#list/path=%2F) and save it in the `/data/` directory.

generate the data.pkl file
```
python preprocess_cicc.py --data=data/data.pkl 
```




train
```
python -u train.py --data=data/data.pkl --from_begin --device=cuda:7 --epochs=50 --drop_rate=0.4 --weight_decay=1e-8 --batch_size=5 --learning_rate=0.0003
```



test
```
python test.py --data data/data.pkl --model_file save/model.pt
```



Finally, The `test_data.json` file with the predicted labels will be saved in the current directory.
