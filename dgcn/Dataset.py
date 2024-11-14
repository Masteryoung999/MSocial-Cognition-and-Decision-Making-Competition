import math
import random
import numpy as np
import torch

class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'A': 0, 'B': 1}
        # 改进后的 label_to_idx 字典，确保没有标签或无效标签时使用 'unknown'
        self.label_to_idx = {
            'Happy': 0,
            'Surprise': 1,
            'Sad': 2,
            'Anger': 3,
            'Disgust': 4,
            'Fear': 5,
            'Neutral': 6,
            'unknown': -1  # test使用 'unknown' 作为默认标签，避免空字符串 ''
        }

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def count(self):
        count = np.zeros(7)
        for s in self.samples:
            for c in s.label:
                # 如果标签为空，跳过此标签
                count[self.label_to_idx.get(c, self.label_to_idx['unknown'])] += 1
        print(count / sum(count))

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()

        mx = torch.max(text_len_tensor).item()
        text_tensor = torch.zeros((batch_size, mx, 768))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            tmp = [torch.from_numpy(t).float().reshape(768) for t in s.text]
            tmp = torch.stack(tmp)
            text_tensor[i, :cur_len, :] = tmp
            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
            
            # 处理标签并填充，避免空标签
            label = [self.label_to_idx.get(c, self.label_to_idx['unknown']) for c in s.label]  # 使用 .get 避免KeyError
            labels.extend(label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)
