#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, subprocess, os
from itertools import combinations, chain
from argparse import ArgumentParser
from os.path import expanduser, join, splitext, normpath, basename

############################
#image size, optimizer, save/print frequency & quality adam
#or lbfgs, adam is faster, lbfgs is higher quality
PRINT_ITER = 50
ITER_COUNT = 150
SAVE_ITER = ITER_COUNT
IMAGE_SIZE = 250
SEED = 123


TV_WEIGHT = 0.0001
CONTENT_WEIGHT = 5
STYLE_WEIGHT = 100
STYLE_SCALE = 1
INIT_SOURCE = "image"

GPU = -1
OPTIMIZER = "adam"
BACKEND = "cudnn"

NEURAL_STYLE_PATH = "Documents/code/zoid-neural-style/"
RUN_SCRIPT_NAME = "neural_style.lua"
MODEL_FILE = "models/VGG_ILSVRC_19_layers.caffemodel"
PROTO_FILE = "models/VGG_ILSVRC_19_layers_deploy.prototxt"
LAYERS_CONTENT = ["relu4_2"]
''' All styles:
LAYERS_STYLE = ["relu1_1", "relu1_2",
                "relu2_1", "relu2_2",
                "relu3_1", "relu3_2", "relu3_3", "relu3_4",
                "relu4_1", "relu4_2", "relu4_3", "relu4_4",
                "relu5_1", "relu5_2", "relu5_3", "relu5_4"]
'''
LAYERS_STYLE = ["relu1_1",
                "relu2_2",
                "relu3_3",
                "relu4_4",
                "relu5_4"]

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--style', type=str,
                        dest='input_style',
                        help='file with input style',
                        metavar='STYLE', required=True)

    parser.add_argument('--in-file', type=str,
                        dest='in_file', help='file to transform',
                        metavar='IN_PATH', required=True)

    parser.add_argument('--out-path', type=str,
                        dest='out_path', help="output directory", metavar='OUT_PATH',
                        required=True)

    return parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    lst = list(iterable)
    return chain.from_iterable(combinations(lst, r) for r in range(len(lst)+1))


# Print iterations progress
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    parser = build_parser()
    options = parser.parse_args()
    '''
    TBD th neural_style.lua \
 1 -init image \
-style_image ../in/pic/src.jpg -content_image ../in/pic/dst4.jpg \
-output_image ../in/pic/outdst4d.jpg \
-image_size 1000 -content_weight 0.000001 -style_weight 100000 \
-save_iter 50 -num_iterations 1000 \
-model_file models/nin_imagenet_conv.caffemodel \
-proto_file models/train_val.prototxt \
-content_layers relu0,relu1,relu2,relu3,relu5,relu6,relu7,relu8,relu9,relu10 \
-style_layers relu0,relu1,relu2,relu3,relu5,relu6,relu7,relu8,relu9,relu10
    '''
    home = expanduser("~")
    print('The home directory is {0}'.format(home))

    output_dir = expanduser(join(options.out_path,
                                 splitext(basename(options.input_style))[0],
                                 splitext(basename(options.in_file))[0]))
    print('The output directory is {0}'.format(output_dir))

    scriptPath = normpath(join(home, NEURAL_STYLE_PATH, RUN_SCRIPT_NAME))
    print('The script path is {0}'.format(scriptPath))

    styleFile = expanduser(options.input_style)
    print('The style path is {0}'.format(styleFile))

    inputFile = expanduser(options.in_file)
    print('The style path is {0}'.format(inputFile))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    styles = list(powerset(LAYERS_STYLE))
    stylesCount = len(styles)
    print('Using {0} pairs of styles'.format(stylesCount))

    i = -1
    for style in styles:
        i = i + 1
        if len(style) > 0:
            print('Style layers: {0}'.format(".".join(style)))

            runScript = 'th {script} -style_scale {style_scale} -init {init} -style_image "{style_image}" \
-content_image "{content_image}" -image_size {image_size} -output_image "{output_image}" \
-content_weight {content_weight} -style_weight {style_weight} -save_iter {save_iter} \
-num_iterations {num_iterations} -content_layers {content_layers} -style_layers {style_layers} \
-gpu {gpu} -optimizer {optimizer} -tv_weight {tv_weight} -backend {backend} -seed {seed} -normalize_gradients'.format(
        script=scriptPath,
        style_scale=STYLE_SCALE,
        init=INIT_SOURCE,
        style_image=styleFile,
        content_image=inputFile,
        image_size=IMAGE_SIZE,
        output_image=join(output_dir, '{0}.jpg'.format(".".join(style))),
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        save_iter=SAVE_ITER,
        num_iterations=ITER_COUNT,
        content_layers=",".join(LAYERS_CONTENT),
        style_layers=",".join(style),
        gpu=GPU,
        optimizer=OPTIMIZER,
        tv_weight=TV_WEIGHT,
        backend=BACKEND,
        seed=SEED)

            #print('Script \'{0}\''.format(runScript))
            printProgress(i, stylesCount, prefix='Progress:',
                          suffix='Complete', barLength=100)

            with subprocess.Popen(runScript, shell=True,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True,
                                  cwd=join(home, NEURAL_STYLE_PATH)) as proc:
                print(proc.stdout.read())

if __name__ == '__main__':
    main()
