
import sys
from argparse import ArgumentParser


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
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions',
                        help='allow different image dimensions')

    return parser
