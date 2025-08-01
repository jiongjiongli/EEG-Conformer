import random
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             f1_score,
                             precision_score,
                             recall_score,
                             precision_recall_curve,
                             accuracy_score)

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_file_infos(data_root_path):
    csv_file_paths = data_root_path.glob("*.csv")
    csv_file_paths = list(csv_file_paths)
    csv_file_paths.sort()

    data_file_infos = {}

    for csv_file_path in csv_file_paths:
        file_name = csv_file_path.stem
        file_name_parts = file_name.split("_", maxsplit=1)

        assert len(file_name_parts) in [2], file_name_parts

        subject = file_name_parts[0][:-1]
        split_type = file_name_parts[0][-1]

        assert split_type in ["E", "T"], split_type

        if split_type == "E":
            split = "test"
        else:
            assert split_type == "T", split_type
            split = "trainval"

        data_file_infos.setdefault(subject, {})

        data_file_infos[subject].setdefault(split, {})

        split_data_file_infos = data_file_infos[subject][split]

        assert file_name_parts[1] in ["eeg_data", "event_position", "event_type", "event_duration", "labels", "eeg_seqs", "filtered_eeg_seqs"], file_name_parts

        assert file_name_parts[1] not in split_data_file_infos, file_name_parts
        split_data_file_infos[file_name_parts[1]] = csv_file_path

    return data_file_infos


def get_dataloaders(subject,
                    seed,
                    data_file_infos,
                    batch_size,
                    val_size,
                    num_channels,
                    seq_len):
    dataset_infos = {}

    for split in ["trainval", "test"]:
        assert split in ["trainval", "test"], split
        split_data_file_infos = data_file_infos[subject][split]

        eeg_data_path = split_data_file_infos["filtered_eeg_seqs"]
        labels_path = split_data_file_infos["labels"]
        eeg_data_df = pd.read_csv(eeg_data_path)
        eeg_data_df = eeg_data_df.drop(columns="Unnamed: 0")
        eeg_data_df = eeg_data_df.fillna(0)
        # shape = [B, seq_len=1K, num_channels=22]
        eeg_data = eeg_data_df.to_numpy().reshape((-1, seq_len, num_channels))
        # shape = [B, num_channels=22, seq_len=1K]
        eeg_data = eeg_data.transpose((0, 2, 1))
        # shape = [B, 1, num_channels=22, seq_len=1K]
        eeg_data = eeg_data[:, np.newaxis]

        labels_df = pd.read_csv(labels_path, names=["label"])
        labels = labels_df["label"].to_numpy() - 1

        assert len(eeg_data) == len(labels), (len(eeg_data), len(labels))

        if split in ["trainval"]:
            eeg_train, eeg_val, labels_train, labels_val = train_test_split(
                eeg_data,
                labels,
                test_size=val_size,
                random_state=seed)

            dataset_infos["train"] = {
                "eeg_data": eeg_train,
                "labels": labels_train,
            }

            dataset_infos["val"] = {
                "eeg_data": eeg_val,
                "labels": labels_val,
            }
        else:
            assert split in ["test"], split

            dataset_infos[split] = {
                "eeg_data": eeg_data,
                "labels": labels,
            }

    datasets = {}
    aug_dataset_infos = {}

    train_data_mean = np.mean(dataset_infos["train"]["eeg_data"])
    train_data_std = np.std(dataset_infos["train"]["eeg_data"])

    for split in dataset_infos:
        eeg_data = dataset_infos[split]["eeg_data"]
        labels   = dataset_infos[split]["labels"]

        norm_eeg_data = (eeg_data - train_data_mean) / train_data_std

        inputs = torch.from_numpy(norm_eeg_data).to(torch.float32)
        gt_labels = torch.from_numpy(labels).to(torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, gt_labels)
        datasets[split] = dataset

        if split in ["train"]:
            aug_dataset_infos["train"] = {
                "aug_inputs": norm_eeg_data,
                "aug_labels": labels,
            }

    dataloaders = {}

    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = torch.utils.data.DataLoader(dataset=datasets[split],
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)

    return dataset_infos, dataloaders, aug_dataset_infos



class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, num_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, num_classes)
        )


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        # input.shape = [B, 1, num_channels=22, seq_len=1000]
        self.shallownet = nn.Sequential(
            # Conv2d: 1 -> 40, K=(1, 25)
            # output.shape = [B, 40, 22, 1000 - 25 + 1]
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            # Conv2d: 40 -> 40, K=(22, 1)
            # output.shape = [B, 40, 1, 1000 - 25 + 1]
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # AvgPool2d: K=(1, 75), S=(1, 15)
            # output.shape = [B, 40, 1, 61=(1000 - 25 + 1 - 75) // 15 + 1]
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            # Conv2d: 40 -> emb_size=40
            #  input.shape = [B, emb_size=40, H=1, W=61]
            # output.shape = [B, H * W=61, emb_size=40]
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        # x.shape = [B, 1, num_channels=22, seq_len=1000]
        b, _, _, _ = x.shape
        # output.shape = [B, 40, 1, 61=(1000 - 25 + 1 - 75) // 15 + 1]
        x = self.shallownet(x)
        # output.shape = [B, H * W=61, emb_size=40]
        x = self.projection(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            # num_classes=4
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        #  input.shape = [B, num_anchors=61, emb_size=40]
        # output.shape = [B, C=num_anchors * emb_size=61*40=2440]
        x = x.contiguous().view(x.size(0), -1)
        #  input.shape = [B, C=num_anchors * emb_size=61*40=2440]
        #  output.shape = [B, C=num_classes=4]
        out = self.fc(x)
        return out



class Trainer:
    def __init__(self,
                 subject,
                 model,
                 device,
                 dataloaders,
                 aug_dataset_infos,
                 batch_size,
                 class_names,
                 num_classes,
                 num_channels,
                 seq_len,
                 num_segments=8,
                 epochs=2000,
                 eval_interval=10,
                 ):
        self.subject = subject
        self.model = model
        self.device = device

        self.model.to(device)

        self.dataloaders = dataloaders
        self.aug_dataset_infos=aug_dataset_infos

        self.batch_size = batch_size

        self.class_names = class_names
        self.num_classes=num_classes

        assert self.batch_size % self.num_classes == 0, (self.batch_size, self.num_classes)

        self.num_channels = num_channels
        self.seq_len = seq_len
        self.num_segments = num_segments
        assert self.seq_len % self.num_segments == 0, (self.seq_len, self.num_segments)

        self.epochs = epochs
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.lr,
                                          betas=(self.b1, self.b2))
        self.criterion = torch.nn.CrossEntropyLoss()

        self.eval_interval = eval_interval

    def train(self):
        train_metrics = []
        val_metrics = []
        test_metrics = []
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_val_acc_epoch = 0

        for epoch_index in range(self.epochs):
            epoch_train_metrics = self.train_one_epoch(epoch_index)

            if epoch_index % self.eval_interval == 0:
                epoch_val_metrics  = self.eval_one_epoch(epoch_index, split="val")
                epoch_test_metrics = self.eval_one_epoch(epoch_index, split="test")

                train_metrics.append(epoch_train_metrics)
                val_metrics.append(epoch_val_metrics)
                test_metrics.append(epoch_test_metrics)

                val_acc = epoch_val_metrics["accuracy"]
                test_acc = epoch_test_metrics["accuracy"]

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_val_acc_epoch = epoch_index

                    torch.save(self.model.state_dict(), 'best_model.pth')


                print(f"[Eval] {self.subject} Epoch {epoch_index:05d} val_acc: {val_acc:.2%} best_val_acc: {best_val_acc:.2%} test_acc: {test_acc:.2%} best_test_acc: {best_test_acc:.2%} best_epoch: {best_val_acc_epoch:05d}")

        train_metrics_df = pd.DataFrame(train_metrics)
        train_metrics_df.to_csv(f"{self.subject}_train_metrics.csv")

        val_metrics_df = pd.DataFrame(val_metrics)
        val_metrics_df.to_csv(f"{self.subject}_val_metrics.csv")

        test_metrics_df = pd.DataFrame(test_metrics)
        test_metrics_df.to_csv(f"{self.subject}_test_metrics.csv")

        print("Completed!")

    def train_one_epoch(self, epoch_index):
        self.model.train()

        running_loss = 0.0
        avg_loss = 0.0
        running_accuracy = 0.0
        avg_accuracy = 0.0

        train_dataloader = self.get_dataloader("train")
        pbar = tqdm(train_dataloader)

        for batch_index, data in enumerate(pbar):
            inputs, labels = data

            # Data Augmentation
            aug_inputs, aug_labels = self.interaug(self.aug_dataset_infos)
            inputs = torch.cat((inputs, aug_inputs))
            labels = torch.cat((labels, aug_labels))

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (1 + batch_index)

            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(prob, dim=-1)

            y_pred = predicted.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()

            # cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            running_accuracy += acc
            avg_accuracy = running_accuracy / (1 + batch_index)

            pbar.set_description(f"[Train] Epoch {epoch_index:05d} Batch {batch_index:05d}")
            pbar.set_postfix(loss=f"{avg_loss:.2f}", acc=f"{avg_accuracy:.2%}")

        train_metrics = dict(
            epoch=epoch_index,
            loss=avg_loss,
            accuracy=avg_accuracy,
        )

        return train_metrics

    def eval_one_epoch(self, epoch_index, split):
        self.model.eval()

        running_loss = 0.0
        avg_loss = 0.0

        all_preds = []
        all_labels = []
        all_scores = []

        test_dataloader = self.get_dataloader(split)
        pbar = tqdm(test_dataloader)

        with torch.no_grad():
            for batch_index, data in enumerate(pbar):
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                avg_loss = running_loss / (1 + batch_index)

                prob = torch.softmax(outputs, dim=-1)
                predict_score, predict_class = torch.max(prob, dim=-1)

                y_score = prob.detach().cpu().numpy()
                y_pred  = predict_class.detach().cpu().numpy()
                y_true  = labels.detach().cpu().numpy()

                all_scores.extend(y_score)
                all_preds.extend(y_pred)
                all_labels.extend(y_true)

                pbar.set_description(f"[{split}] Epoch {epoch_index:05d} Batch {batch_index:05d}")

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        test_metrics = self.get_metrics(epoch_index,
                                        split,
                                        avg_loss,
                                        all_labels,
                                        all_preds,
                                        all_scores,
                                        )

        return test_metrics

    def get_metrics(self,
                    epoch_index,
                    split,
                    avg_loss,
                    y_true,
                    y_pred,
                    y_score):
        class_labels = np.arange(self.num_classes)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=class_labels, average=None)
        recall    =    recall_score(y_true, y_pred, labels=class_labels, average=None)
        f1        =        f1_score(y_true, y_pred, labels=class_labels, average=None)

        metrics = dict(
            epoch=epoch_index,
            loss=avg_loss,
            accuracy=acc,
            # confusion_matrix=cm,
            precision=precision,
            recall=recall,
            f1=f1
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.class_names)
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.subject}_{split}_epoch_{epoch_index:05d}_cm.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))

        for class_idx, class_name in enumerate(self.class_names):
            # Binarize true labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_score_class = y_score[:, class_idx]  # Probabilities for this class

            class_precision, class_recall, thresholds = precision_recall_curve(y_true_binary, y_score_class)

            # print(f"Class {class_idx}: Precision-Recall curve has {len(thresholds)} points")
            plt.plot(class_recall, class_precision, label=class_name)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.subject}_{split}_epoch_{epoch_index:05d}_pr.png")
        plt.close(fig)

        return metrics

    def get_dataloader(self, split):
        return self.dataloaders[split]

    def interaug(self, aug_dataset_infos):
        aug_dataset_info = aug_dataset_infos["train"]
        aug_inputs = aug_dataset_info["aug_inputs"]
        aug_labels = aug_dataset_info["aug_labels"]

        num_class_samples = self.batch_size // self.num_classes
        segment_len = self.seq_len // self.num_segments

        aug_data = []
        aug_label = []
        for gt_label in range(self.num_classes):
            cls_idx = np.nonzero(aug_labels == gt_label)
            tmp_data  = aug_inputs[cls_idx]


            tmp_aug_data = np.zeros((num_class_samples,
                                     1,
                                     self.num_channels,
                                     self.seq_len))
            for ri in range(num_class_samples):
                rand_idx = np.random.randint(0, tmp_data.shape[0], self.num_segments)
                for rj in range(self.num_segments):
                    tmp_aug_data[ri, :, :, rj * segment_len:(rj + 1) * segment_len] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * segment_len:(rj + 1) * segment_len]
            tmp_aug_label = np.full(num_class_samples,
                                    fill_value=gt_label,
                                    dtype=aug_labels.dtype)

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_aug_label)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        aug_inputs = torch.from_numpy(aug_data).to(torch.float32)
        aug_labels = torch.from_numpy(aug_label).to(torch.long)
        return aug_inputs, aug_labels


data_root_paths = (
    r"/content/drive/MyDrive/data/bci",
    r"/home/iconsense/Desktop/jiongjiong_li/proj/experiment/BCICIV_2a_csv",
)

def find_dir_path(dir_paths):
    for dir_path in dir_paths:
        dir_path = Path(dir_path)
        if dir_path.exists():
            return dir_path

    return None

data_root_path = find_dir_path(data_root_paths)

seed = 17
num_channels = 22
seq_len = 1000
batch_size = 72
class_names = ['Left hand', 'Right hand', 'Both feet', 'Tongue']
num_classes = len(class_names)

val_size = 0.1
num_segments=8
epochs=2000

eval_interval=10

subject_ids = range(4, 9 + 1)
subject_ids = list(subject_ids)

for subject_id in subject_ids:
    subject = f"A0{subject_id}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)

    data_file_infos = get_data_file_infos(data_root_path)

    dataset_infos, dataloaders, aug_dataset_infos = get_dataloaders(
        subject,
        seed,
        data_file_infos,
        batch_size=batch_size,
        val_size=val_size,
        num_channels=num_channels,
        seq_len=seq_len
    )

    gt_label_counts = []

    for split, dataset_info in dataset_infos.items():
        eeg_data = dataset_info["eeg_data"]
        labels = dataset_info["labels"]

        unique_labels, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique_labels, counts):
            class_name = class_names[label]
            percent = count / len(labels)

            gt_label_counts.append({
                "Split": split,
                "ClassName": class_name,
                "LabelCount": count,
                "Percent": f"{percent:.1%}"
            })

    gt_label_counts_df = pd.DataFrame(gt_label_counts)
    display(gt_label_counts_df)

    model = Conformer(num_classes=num_classes)

    trainer = Trainer(subject,
                      model,
                      device,
                      dataloaders,
                      aug_dataset_infos,
                      batch_size=batch_size,
                      class_names=class_names,
                      num_classes=num_classes,
                      num_channels=num_channels,
                      seq_len=seq_len,
                      num_segments=num_segments,
                      epochs=epochs,
                      eval_interval=eval_interval,
                    )

    trainer.train()
