import argparse
import time
import cv2
import glob
import numpy as np
import os
import torch
from collections import OrderedDict
from basicsr.archs.MSDAN_arch import MSDAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../experiments/pretrained_models/MSDAN_x4.pth')
    parser.add_argument('--input', type=str, default='../datasets/set14/mod4/LRx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/MSDAN', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = MSDAN(channels=48, num_DFEB=8, upscale_factor=4)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    test_results = OrderedDict()
    test_results['runtime'] = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                start.record()
                output = model(img)
                end.record()
                torch.cuda.synchronize()
                test_results['runtime'].append(start.elapsed_time(end))  # milliseconds
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)

    tot_runtime = sum(test_results['runtime']) / 1000.0
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    print('------> Total runtime of ({}) is : {:.6f} seconds = {:.2f} ms'.format(args.input, tot_runtime, tot_runtime * 1000))
    print('------> Average runtime of ({}) is : {:.6f} seconds = {:.2f} ms'.format(args.input, ave_runtime, ave_runtime * 1000))

if __name__ == '__main__':
    main()

