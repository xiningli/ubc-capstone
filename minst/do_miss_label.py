import numpy as np
import random
def do_miss_label(y_train_labels, miss_labeled_ratio):
    y_train_copy = np.copy(y_train_labels)
    all_labels = set(y_train_copy)
    all_labels_count = len(all_labels)
    misslabeled_indices = np.random.choice(len(y_train_copy)-1,
                                           int((len(y_train_copy)-1)*miss_labeled_ratio))
    for misslabeled_index in misslabeled_indices:
        y_train_copy[misslabeled_index] = random.randrange(0,all_labels_count)
    return y_train_copy