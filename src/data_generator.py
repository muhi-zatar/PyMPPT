import os
import random

import numpy as np
from tensorflow.keras.utils import Sequence

from features import Features


class DataGenerator(Sequence):
    def __init__(self, hparams, loss_weights=[], data_type='train'):
        self.batch_size = hparams["batch_size"]
        self.loss_weights = loss_weights
        self.features_type = hparams["features_type"]

        self.data_tasks = [t["name"] for t in hparams["tasks"]]
        self.selected_tasks = hparams["selected_tasks"]
        if "all_tasks" in hparams:
            self.all_tasks = hparams["all_tasks"]
        else:
            self.all_tasks = hparams["selected_tasks"]

        self.task_len = {}
        self.shuffle_tasks = hparams["shuffle_tasks"]

        self.maps = {}
        self.task_multiplier = {}
        for params in hparams["tasks"]:
            class_to_int = {label: index for index, label in enumerate(params["classes"])}
            self.maps[params["name"]] = class_to_int
            self.task_multiplier[params["name"]] = params["multiplier"]

        # Available files are "train", "dev" and "test"
        self.data_type = data_type
        self.data_dir = hparams['data_dir']
        self.filename = os.path.join(self.data_dir, hparams[self.data_type + "_file"])

        self.build_dataset(hparams["selected_datasets"])
        self.reset()

    def reset(self):
        self.indexes = {t: 0 for t in self.selected_tasks}
        for t in self.selected_tasks:
            random.shuffle(self.datasets[t])
            self.task_len[t] = int(np.floor(len(self.datasets[t]) / self.batch_size))

    def build_dataset(self, selected_datasets):
        features = Features.get_features(self.data_dir, self.features_type)
        self.datasets = {t: [] for t in self.all_tasks}
        with open(self.filename) as f:
            f.readline()
            for line in f:
                values = line.strip().split()
                utt_id = values[0]
                dataset = values[1].strip().lower()
                raw_item = {k: v.lower() for k, v in zip(self.data_tasks, values[3:])}
                if utt_id in features:
                    if dataset in selected_datasets:
                        self._add_record(raw_item, features[utt_id], dataset)
                else:
                    print("Utterance does not have features!!! utt_id: {}".format(utt_id))

    def _add_record(self, raw_item, features, dataset):
        for task in self.all_tasks:
            if raw_item[task] in self.maps[task]:
                multiplier = self.task_multiplier[task] if self.data_type == 'train' else 1
                for _ in range(multiplier):
                    self.datasets[task].append(
                        [features, self.maps[task][raw_item[task]], dataset])

    def __len__(self):
        return sum(self.task_len.values())

    def get_inputs(self, batch):
        X_batch = [x[0] for x in batch]
        if self.features_type in ["mfcc", "mel"]:
            max_len = max(x.shape[1] for x in X_batch)

            # Add padding to sequences
            X_batch_tmp = []
            for sequence in X_batch:
                pad_len = max_len - sequence.shape[1] + 1
                pad = np.zeros((sequence.shape[0], pad_len), dtype=np.float32)
                padded_signal = np.concatenate((pad, sequence), axis=1)
                X_batch_tmp.append(padded_signal)

            X_batch = np.transpose(np.asarray(X_batch_tmp), (0, 2, 1))
        else:
            X_batch = np.asarray(X_batch)
        return X_batch

    def __getitem__(self, idx):
        if self.shuffle_tasks:
            weights = [self.task_len[t] - self.indexes[t] + 1 for t in self.selected_tasks]
            task = random.choices(self.selected_tasks, weights=weights)[0]
        else:
            for t in self.selected_tasks:
                if self.indexes[t] < self.task_len[t]:
                    task = t
                    break
        index = self.indexes[task]
        batch = self.datasets[task][index*self.batch_size:(index+1)*self.batch_size]
        self.indexes[task] += 1

        # Prepare outputs
        Y_batch = []
        for i, t in enumerate(self.all_tasks):
            if t == task:
                Y_batch.append(np.asarray([x[1] for x in batch], dtype=np.int32))
                self.loss_weights[i].assign(1)
            else:
                Y_batch.append(np.zeros(len(batch), dtype=np.int32))
                self.loss_weights[i].assign(0)

        # Prepare inputs
        X_batch = self.get_inputs(batch)

        return X_batch, Y_batch
