import math
import numpy as np
from parse import args


def load_file(file_path):
    m_item, all_pos = 0, []

    with open(file_path, "r") as f:
        for line in f.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)

    return m_item, all_pos


def load_dataset(path):
    m_item = 0
    m_item_, all_train_ind = load_file(path + "train.txt")
    m_item = max(m_item, m_item_)
    m_item_, all_test_ind = load_file(path + "test.txt")
    m_item = max(m_item, m_item_)

    if args.part_percent > 0:
        _, part_train_ind = load_file(path + "train.part-{}%.txt".format(args.part_percent))
    else:
        part_train_ind = []

    items_popularity = np.zeros(m_item)
    for items in all_train_ind:
        for item in items:
            items_popularity[item] += 1
    for items in all_test_ind:
        for item in items:
            items_popularity[item] += 1

    return m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity


def sample_part_of_dataset(path, ratio):
    # sample in the whole dataset instead of each user
    m_items, all_pos = load_file(path + "train.txt")
    user_interacted = [[] for _ in range(len(all_pos))]
    num_inter = sum([len(pos) for pos in all_pos])
    with open(path + "train.part-{}%.txt".format(int(ratio * 100)), "w") as f:
        num_sample = math.ceil(num_inter * ratio)
        sampled = 0
        while sampled < num_sample:
            user = np.random.randint(len(all_pos))
            if len(all_pos[user]) == 0:
                continue
            item = np.random.choice(all_pos[user])
            if item not in user_interacted[user]:
                user_interacted[user].append(item)
                sampled += 1
        for user, pos_items in enumerate(user_interacted):
            f.write(str(user) + ' ')
            f.write(' '.join([str(item) for item in pos_items]))
            f.write('\n')