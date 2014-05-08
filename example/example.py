#!/usr/bin/env python
import numpy as np

from misvmio import parse_c45, bag_set
import misvm


def main():
    # Load list of C4.5 Examples
    example_set = parse_c45('musk1')

    # Group examples into bags
    bagset = bag_set(example_set)

    # Convert bags to NumPy arrays
    # (The ...[:, 2:-1] removes first two columns and last column,
    #  which are the bag/instance ids and class label)
    bags = [np.array(b.to_float())[:, 2:-1] for b in bagset]
    labels = np.array([b.label for b in bagset], dtype=float)
    # Convert 0/1 labels to -1/1 labels
    labels = 2 * labels - 1

    # Spilt dataset arbitrarily to train/test sets
    train_bags = bags[10:]
    train_labels = labels[10:]
    test_bags = bags[:10]
    test_labels = labels[:10]

    # Construct classifiers
    classifiers = {}
    classifiers['MissSVM'] = misvm.MissSVM(kernel='linear', C=1.0, max_iters=10)
    classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
    classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)

    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in classifiers.items():
        classifier.fit(train_bags, train_labels)
        predictions = classifier.predict(test_bags)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))

    for algorithm, accuracy in accuracies.items():
        print '\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy)


if __name__ == '__main__':
    main()
