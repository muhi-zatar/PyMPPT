import os
import pickle

import numpy as np


class Features:
    __data = None
    __type = None

    @staticmethod
    def get_features(data_dir, features_type):
        """ Static access method. """
        if features_type != Features.__type:
            Features.__data = None

        if Features.__data is None:
            Features(data_dir, features_type)
            Features.__type = features_type

        return Features.__data

    def __init__(self, data_dir, features_type):
        if Features.__data is not None:
            raise Exception("This class is a singleton!")
        else:
            if "vector" in features_type:
                vector_types = features_type.replace("vector", "")
                features = None
                for feat in ["i", "d", "x"]:
                    if feat not in vector_types:
                        continue

                    feat_path = os.path.join(data_dir, feat + "vector.pickle")
                    with open(feat_path, 'rb') as f:
                        data = pickle.load(f)
                        if features:
                            new_feats = {k: np.concatenate((features[k], data[k]))
                                         for k in data.keys() if k in features}
                            features = new_feats
                        else:
                            features = data

                Features.__data = features
            else:
                features_path = os.path.join(data_dir, features_type + ".pickle")
                with open(features_path, 'rb') as f:
                    Features.__data = pickle.load(f)
