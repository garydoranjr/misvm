"""
Parses and represents C4.5 MI data sets
"""
from __future__ import print_function, division
import os
import re
import sys
import traceback
from collections import MutableSequence, defaultdict, Sequence
from itertools import chain, starmap

NAMES_EXT = '.names'
DATA_EXT = '.data'

_COMMENT_RE = '//.*'
_BINARY_RE = '\\s*0\\s*,\\s*1\\s*'


class Feature(object):
    """
    Information for a feature
    of C4.5 data
    """

    class Type:
        """
        Type of feature
        """
        CLASS = 'CLASS'
        ID = 'ID'
        BINARY = 'BINARY'
        NOMINAL = 'NOMINAL'
        CONTINUOUS = 'CONTINUOUS'

    def __init__(self, name, ftype, values=None):
        self.name = name
        self.type = ftype
        if (self.type == Feature.Type.ID or
                    self.type == Feature.Type.NOMINAL):
            if values is None:
                raise Exception('No values for %s feature' % self.type)
            else:
                self.values = tuple(values)
        else:
            if values is None:
                self.values = None
            else:
                raise Exception('Values given for % feature' % self.type)
        self.tup = (self.name, self.type, self.values)

    def __cmp__(self, other):
        if self.tup > other.tup:
            return 1
        elif self.tup < other.tup:
            return -1
        else:
            return 0

    def __hash__(self):
        return self.tup.__hash__()

    def __repr__(self):
        return '<%s, %s, %s>' % self.tup

    def to_float(self, value):
        if value is None:
            return None
        if (self.type == Feature.Type.ID or
                    self.type == Feature.Type.NOMINAL):
            return float(self.values.index(value))
        elif (self.type == Feature.Type.BINARY or
                      self.type == Feature.Type.CLASS):
            if value:
                return 1.0
            else:
                return 0.0
        else:
            return value


Feature.CLASS = Feature("CLASS", Feature.Type.CLASS)


class Schema(Sequence):
    """
    Represents a schema for C4.5 data
    """

    def __init__(self, features):
        self.features = tuple(features)

    def __cmp__(self, other):
        if self.features > other.features:
            return 1
        elif self.features < other.features:
            return -1
        else:
            return 0

    def __hash__(self):
        return self.features.__hash__()

    def __repr__(self):
        return str(self.features)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __contains__(self, item):
        return self.features.__contains__(item)

    def __getitem__(self, key):
        return self.features[key]


class ExampleSet(MutableSequence):
    """
    Holds a set of examples
    """

    def __init__(self, schema):
        self.schema = schema
        self.examples = []

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self.examples.__iter__()

    def __contains__(self, item):
        return self.examples.__contains__(item)

    def __getitem__(self, key):
        return self.examples[key]

    def __setitem__(self, key, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        self.examples[key] = example

    def __delitem__(self, key):
        del self.examples[key]

    def insert(self, key, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        return self.examples.insert(key, example)

    def append(self, example):
        if example.schema != self.schema:
            raise ValueError('Schema mismatch')
        super(ExampleSet, self).append(example)

    def __repr__(self):
        return '<%s, %s>' % (self.schema, self.examples)

    def to_float(self, normalizer=None):
        return [ex.to_float(normalizer) for ex in self]


class Example(MutableSequence):
    """
    Represents a single example
    from a dataset
    """

    def __init__(self, schema):
        self.schema = schema
        self.features = [None for i in range(len(schema))]
        self.weight = 1.0

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def __contains__(self, item):
        return self.features.__contains__(item)

    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        self.features[key] = value

    def __delitem__(self, key):
        del self.features[key]

    def insert(self, key, item):
        return self.features.insert(key, item)

    def __repr__(self):
        return '<%s, %s, %s>' % (self.schema, self.features, self.weight)

    def copy_of(self):
        ex = Example(self.schema)
        for i, f in enumerate(self):
            ex[i] = f
        return ex

    def to_float(self, normalizer=None):
        if normalizer is None:
            normalizer = lambda x: x
        return normalizer([feature.to_float(value)
                           for feature, value in zip(self.schema, self)])


class Bag(MutableSequence):
    """
    Represents a Bag
    """

    def __init__(self, bag_id, examples):
        classes = map(lambda x: x[-1], examples)
        if any(classes):
            self.label = True
        else:
            self.label = False
        self.bag_id = bag_id
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self.examples.__iter__()

    def __contains__(self, item):
        return self.examples.__contains__(item)

    def __getitem__(self, key):
        return self.examples[key]

    def __setitem__(self, key, value):
        self.examples[key] = value

    def __delitem__(self, key):
        del self.examples[key]

    def insert(self, key, item):
        return self.examples.insert(key, item)

    def __repr__(self):
        return '<%s, %s>' % (self.examples, self.label)

    def to_float(self, normalizer=None):
        return [example.to_float(normalizer) for example in self]


def bag_set(exampleset, bag_attr=0):
    """
    Construct bags on the given attribute
    """
    bag_dict = defaultdict(list)
    for example in exampleset:
        bag_dict[example[bag_attr]].append(example)
    return [Bag(bag_id, value) for bag_id, value in bag_dict.items()]


def parse_c45(file_base, rootdir='.'):
    """
    Returns an ExampleSet from the
    C4.5 formatted data
    """
    schema_name = file_base + NAMES_EXT
    data_name = file_base + DATA_EXT
    schema_file = find_file(schema_name, rootdir)
    if schema_file is None:
        raise ValueError('Schema file not found')
    data_file = find_file(data_name, rootdir)
    if data_file is None:
        raise ValueError('Data file not found')
    return _parse_c45(schema_file, data_file)


def _parse_c45(schema_filename, data_filename):
    """Parses C4.5 given file names"""
    try:
        schema = _parse_schema(schema_filename)
    except Exception as e:
        raise Exception('Error parsing schema: %s' % e)

    try:
        examples = _parse_examples(schema, data_filename)
    except Exception as e:
        raise Exception('Error parsing examples: %s' % e)

    return examples


def _parse_schema(schema_filename):
    features = []
    needs_id = True
    with open(schema_filename) as schema_file:
        for line in schema_file:
            feature = _parse_feature(line, needs_id)
            if feature is not None:
                if (needs_id and
                            feature.type == Feature.Type.ID):
                    needs_id = False
                features.append(feature)
    try:
        features.remove(Feature.CLASS)
    except:
        raise Exception('File does not contain worthless "Class" line.')
    features.append(Feature.CLASS)
    return Schema(features)


def _parse_feature(line, needs_id):
    """
    Parse a feature from the given line;
    second argument indicates whether we
    need an ID for our schema
    """
    line = _trim_line(line)
    if len(line) == 0:
        # Blank line
        return None
    if re.match(_BINARY_RE, line) is not None:
        # Class feature
        return Feature.CLASS
    colon = line.find(':')
    if colon < 0:
        raise Exception('No feature name found.')
    name = line[:colon].strip()
    remainder = line[colon + 1:]
    values = _parse_values(remainder)
    if needs_id:
        return Feature(name, Feature.Type.ID, values)
    elif len(values) == 1 and values[0].startswith('continuous'):
        return Feature(name, Feature.Type.CONTINUOUS)
    elif len(values) == 2 and '0' in values and '1' in values:
        return Feature(name, Feature.Type.BINARY)
    else:
        return Feature(name, Feature.Type.NOMINAL, values)


def _parse_values(remainder):
    values = list()
    for raw in remainder.split(','):
        raw = raw.strip()
        if len(raw) > 1 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1].strip()
        values.append(raw)
    return values


def _parse_examples(schema, data_filename):
    exset = ExampleSet(schema)
    with open(data_filename) as data_file:
        for line in data_file:
            line = _trim_line(line)
            if len(line) == 0:
                continue
            try:
                ex = _parse_example(schema, line)
                exset.append(ex)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                print('Warning: skipping line: "%s"' % line, file=sys.stderr)
    return exset


def _parse_example(schema, line):
    values = _parse_values(line)
    if len(values) != len(schema):
        raise Exception('Feature-data size mismatch: %s' % line)
    ex = Example(schema)
    for i, value in enumerate(values):
        if value == '?':
            # Unknown value says 'None'
            continue
        stype = schema[i].type
        if (stype == Feature.Type.ID or
                    stype == Feature.Type.NOMINAL):
            ex[i] = value
        elif (stype == Feature.Type.BINARY or
                      stype == Feature.Type.CLASS):
            ex[i] = bool(int(value))
        elif stype == Feature.Type.CONTINUOUS:
            ex[i] = float(value)
        else:
            raise ValueError('Unknown schema type "%s"' % stype)
    return ex


def _trim_line(line):
    """
    Removes comments and periods
    from the given line
    """
    line = re.sub(_COMMENT_RE, '', line)
    line = line.strip()
    if len(line) > 0 and line[-1] == '.':
        line = line[:-1].strip()
    return line


def find_file(filename, rootdir):
    """
    Finds a file with filename located in
    some subdirectory of the current directory
    """
    for dirpath, _, filenames in os.walk(rootdir):
        if filename in filenames:
            return os.path.join(dirpath, filename)


def save_c45(example_set, basename, basedir='.'):
    schema_name = os.path.join(basedir, basename + NAMES_EXT)
    data_name = os.path.join(basedir, basename + DATA_EXT)

    print(schema_name)
    with open(schema_name, 'w+') as schema_file:
        schema_file.write('0,1.\n')
        for feature in example_set.schema:
            if (feature.type == Feature.Type.ID or
                        feature.type == Feature.Type.NOMINAL):
                schema_file.write('%s:%s.\n' %
                                  (feature.name, ','.join(sorted(feature.values))))
            elif feature.type == Feature.Type.BINARY:
                schema_file.write('%s:0,1.\n' % feature.name)
            elif feature.type == Feature.Type.CONTINUOUS:
                schema_file.write('%s:continuous.\n' % feature.name)

    with open(data_name, 'w+') as data_file:
        for example in example_set:
            ex_strs = starmap(_feature_to_str, zip(example.schema, example))
            data_file.write('%s.\n' % ','.join(ex_strs))


def _feature_to_str(feature, value):
    if (feature.type == Feature.Type.ID or
                feature.type == Feature.Type.NOMINAL):
        return value
    elif (feature.type == Feature.Type.BINARY or
                  feature.type == Feature.Type.CLASS):
        return str(int(value))
    elif feature.type == Feature.Type.CONTINUOUS:
        return str(float(value))
