import codecs
import json
import os
import pickle
import random
import struct
import sys

import numpy


FEATURE_VECTOR_SIZE = 3


def generate_first_silence(n):
    return [[random.random(), random.random(), random.random()] for _ in range(n)]


def generate_second_silence(n):
    return [[random.random() + 1.0, random.random() + 1.0, random.random() + 1.0] for _ in range(n)]

def generate_first_sound():
    n1 = random.randint(1, 10)
    n2 = random.randint(1, 2)
    n3 = random.randint(4, 7)
    n4 = random.randint(1, 3)
    n5 = random.randint(1, 10)
    is_first_silence = (random.random() >= 0.5)
    if is_first_silence:
        res = generate_first_silence(n1)
    else:
        res = generate_second_silence(n1)
    res += [[2.0 + random.random(), 4.0 + random.random(), 2.0 + random.random()] for _ in range(n2)]
    res += [[6.0 + random.random(), 4.0 + random.random(), 2.0 + random.random()] for _ in range(n3)]
    res += [[6.0 + random.random(), 5.0 + random.random(), 7.0 + random.random()] for _ in range(n4)]
    res += generate_first_silence(n5) if is_first_silence else generate_second_silence(n5)
    return res


def generate_second_sound():
    n1 = random.randint(1, 10)
    n2 = random.randint(1, 4)
    n3 = random.randint(3, 7)
    n4 = random.randint(1, 10)
    is_first_silence = (random.random() >= 0.5)
    if is_first_silence:
        res = generate_first_silence(n1)
    else:
        res = generate_second_silence(n1)
    res += [[2.0 + random.random(), 2.0 + random.random(), 2.0 + random.random()] for _ in range(n2)]
    res += [[4.0 + random.random(), 2.0 + random.random(), 5.0 + random.random()] for _ in range(n3)]
    res += generate_first_silence(n4) if is_first_silence else generate_second_silence(n4)
    return res


def generate_third_sound():
    n1 = random.randint(1, 10)
    n2 = random.randint(2, 3)
    n3 = random.randint(2, 4)
    n4 = random.randint(4, 6)
    n5 = random.randint(1, 2)
    n6 = random.randint(1, 10)
    is_first_silence = (random.random() >= 0.5)
    if is_first_silence:
        res = generate_first_silence(n1)
    else:
        res = generate_second_silence(n1)
    res += [[3.0 + random.random(), 5.0 + random.random(), 7.0 + random.random()] for _ in range(n2)]
    res += [[10.0 + random.random(), 10.0 + random.random(), 10.0 + random.random()] for _ in range(n3)]
    res += [[7.0 + random.random(), 4.0 + random.random(), 2.0 + random.random()] for _ in range(n4)]
    res += [[2.0 + random.random(), 6.0 + random.random(), 3.0 + random.random()] for _ in range(n5)]
    res += generate_first_silence(n6) if is_first_silence else generate_second_silence(n6)
    return res


def save_spectrogram(filename, spectrogram):
    with open(filename, 'wb') as fp:
        pickle.dump(numpy.array(spectrogram, dtype=numpy.float32), fp)
    with open(filename + '.bin', 'wb') as fp:
        spectrogram_size = len(spectrogram)
        fp.write(spectrogram_size.to_bytes(4, byteorder=sys.byteorder, signed=True))
        fp.write(FEATURE_VECTOR_SIZE.to_bytes(4, byteorder=sys.byteorder, signed=True))
        for spectrum_ind in range(spectrogram_size):
            for freq_ind in range(FEATURE_VECTOR_SIZE):
                fp.write(bytearray(struct.pack("f", spectrogram[spectrum_ind][freq_ind])))


def main():
    if len(sys.argv) > 1:
        dir_name = os.path.normpath(sys.argv[1])
        assert os.path.isdir(dir_name), 'Directory `{0}` does not exist!'.format(dir_name)
    else:
        dir_name = ''
    structure = {
        'train': {
            'silence': [
                os.path.join('_background_noise_', 'silence01.fbanks'),
                os.path.join('_background_noise_', 'silence02.fbanks'),
                os.path.join('_background_noise_', 'silence03.fbanks')
            ],
            'speech': {
                'first': [
                    {'source': os.path.join('first', 'first01.fbanks')},
                    {'source': os.path.join('first', 'first02.fbanks')},
                    {'source': os.path.join('first', 'first03.fbanks')}
                ],
                'second': [
                    {'source': os.path.join('second', 'second01.fbanks')},
                    {'source': os.path.join('second', 'second02.fbanks')},
                    {'source': os.path.join('second', 'second03.fbanks')}
                ],
                'third': [
                    {'source': os.path.join('third', 'third01.fbanks')},
                    {'source': os.path.join('third', 'third02.fbanks')},
                    {'source': os.path.join('third', 'third03.fbanks')}
                ]
            }
        },
        'validation': {
            'silence': [
                os.path.join('_background_noise_', 'silence04.fbanks'),
                os.path.join('_background_noise_', 'silence05.fbanks')
            ],
            'speech': {
                'first': [
                    {'source': os.path.join('first', 'first04.fbanks')},
                    {'source': os.path.join('first', 'first05.fbanks')}
                ],
                'second': [
                    {'source': os.path.join('second', 'second04.fbanks')},
                    {'source': os.path.join('second', 'second05.fbanks')}
                ],
                'third': [
                    {'source': os.path.join('third', 'third04.fbanks')},
                    {'source': os.path.join('third', 'third05.fbanks')}
                ]
            }
        },
        'test': {
            'silence': [
                os.path.join('_background_noise_', 'silence06.fbanks')
            ],
            'speech': {
                'first': [
                    {'source': os.path.join('first', 'first06.fbanks')}
                ],
                'second': [
                    {'source': os.path.join('second', 'second06.fbanks')}
                ],
                'third': [
                    {'source': os.path.join('third', 'third06.fbanks')}
                ]
            }
        }
    }
    basedir = os.path.normpath('testdata')
    if not os.path.isdir(os.path.join(dir_name, basedir)):
        os.mkdir(os.path.join(dir_name, basedir))
    if not os.path.isdir(os.path.join(dir_name, basedir, "first")):
        os.mkdir(os.path.join(dir_name, basedir, "first"))
    if not os.path.isdir(os.path.join(dir_name, basedir, "second")):
        os.mkdir(os.path.join(dir_name, basedir, "second"))
    if not os.path.isdir(os.path.join(dir_name, basedir, "third")):
        os.mkdir(os.path.join(dir_name, basedir, "third"))
    if not os.path.isdir(os.path.join(dir_name, basedir, "_background_noise_")):
        os.mkdir(os.path.join(dir_name, basedir, "_background_noise_"))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence01.fbanks"),
                     generate_first_silence(random.randint(20, 50)))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence02.fbanks"),
                     generate_second_silence(random.randint(20, 50)))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence03.fbanks"),
                     generate_first_silence(random.randint(20, 50)))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence04.fbanks"),
                     generate_second_silence(random.randint(20, 50)))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence05.fbanks"),
                     generate_first_silence(random.randint(20, 50)))
    save_spectrogram(os.path.join(dir_name, basedir, "_background_noise_", "silence06.fbanks"),
                     generate_second_silence(random.randint(20, 50)))
    for ind in range(6):
        save_spectrogram(os.path.join(dir_name, basedir, 'first', 'first{0:0>2}.fbanks'.format(ind + 1)),
                         generate_first_sound())
        save_spectrogram(os.path.join(dir_name, basedir, 'second', 'second{0:0>2}.fbanks'.format(ind + 1)),
                         generate_second_sound())
        save_spectrogram(os.path.join(dir_name, basedir, 'third', 'third{0:0>2}.fbanks'.format(ind + 1)),
                         generate_third_sound())
    with codecs.open(os.path.join(dir_name, 'distribution_test.json'), mode='w', encoding='utf-8',
                     errors='ignore') as fp:
        json.dump(structure, fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()