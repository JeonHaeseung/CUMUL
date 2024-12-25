import itertools
import numpy as np
from tqdm import tqdm


DATA_PATH = "path/to/data.npz"
OUTPUT_PATH = "path/to/feauture.npz"


def load_data(data_path):
    train_data = np.load(data_path)
    x_train, y_train = train_data['data'], train_data['labels']
    return x_train, y_train


# CHANGE: added zero padding removal
def trunc_zero_padding(trace):
    start_zero_padding_index = np.asarray(np.where(trace!=0))[0][-1]
    trace = trace[:(start_zero_padding_index+1)]
    return trace


# CHANGE: changed `Instance` class to npz file
def get_feature(trace):
    # classLabel = label_set.index(website_list[instance_index])   # CHANGE: don't need this
    feature = []
    total = []
    cum = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0

    # Process trace
    for packet in trace: # CHANGE: itertools.islice(instance.packets) -> trace
        packetsize = np.sign(packet) * 512   # CHANGE: int(item.packetsize) -> np.sign(packet) * 512

        # incoming packets
        if packetsize > 0:
            inSize += packetsize
            inCount += 1
            # cumulated packetsizes
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(packetsize)
                pos.append(packetsize)
                neg.append(0)
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + packetsize)
                neg.append(neg[-1] + 0)

        # outgoing packets
        if packetsize < 0:
            outSize += abs(packetsize)
            outCount += 1
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(abs(packetsize))
                pos.append(0)
                neg.append(abs(packetsize))
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(packetsize))

    # add feature
    # feature.append(classLabel)   # CHANGE: don't need this
    feature.append(inCount)
    feature.append(outCount)
    feature.append(outSize)
    feature.append(inSize)

    # CHANGE: added zero padding
    if len(cum) < 100:
        cum = np.concatenate((cum, np.zeros(100 - len(cum), dtype=int)))
    else:
        cum = cum[:100]

    feature.extend(cum)    # CHANGE: added cumul feature
    return feature


def iter_feature(x_train, y_train):
    features = []
    for instance in tqdm(x_train):
        trace = trunc_zero_padding(instance)
        feature = get_feature(trace)
        features.append(feature)
    return features, y_train


def save_npz_data(output_path, features, labels):
    np.savez(output_path, data=features, labels=labels)


def save_csv_data(output_path, features, label):
    x = np.array(features)
    y = np.array(label)

    with open(output_path, "w") as file:
        for label, features in zip(y, x):
            line = f"{label} " + " ".join([f"{i + 1}:{value}" for i, value in enumerate(features)])
            file.write(line + "\n")


def main():
    x_train, y_train = load_data(DATA_PATH)
    features, labels = iter_feature(x_train, y_train)
    save_npz_data(OUTPUT_PATH, features, labels)


if __name__ == "__main__":
    main()