import argparse
import cv2
import glob
import numpy as np
import os
import torch
import time
from basicsr.archs.EFEN_arch import EFEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/EFEN_DF2K_x4_1000k.pth')
    parser.add_argument('--input', type=str, default='datasets/set14/mod4/LRx4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/EFEN', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = EFEN(channels=48, num_DFEB=8, upscale_factor=4)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

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
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)


if __name__ == '__main__':
    time1 = time.time()
    main()
    print(time.time()-time1, 's')
