import os
import os.path as osp
import base64
import io
import argparse

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='input data')
    parser.add_argument('--prepared_input_dir', type=str, help='prepared input data directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data = args.input_data
    prepared_input_dir = args.prepared_input_dir

    data_encoded = data.encode('utf-8')
    print('Data received: {}'.format(data))

    img_bytes = base64.b64decode(data_encoded)    # Convert it into bytes
    im = Image.open(io.BytesIO(img_bytes))    # Convert bytes data to PIL Image object

    os.makedirs(prepared_input_dir, exist_ok=True)
    im.save(osp.join(prepared_input_dir, 'input_image.png'))
