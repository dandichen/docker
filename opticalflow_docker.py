import sys
import requests
import base64
import json
import cv2
import time
import os
from PIL import Image
import StringIO
import fnmatch
import numpy as np

import matplotlib.pyplot as plt

def plot_velocity_vector(flow):

    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - 20, 30):
        for j in range(0, img.shape[1] - 20, 30):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                    (150, 0, 0), 2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                             (150, 0, 0), 2)
                else:
                    cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')


def flow2color(flow):
    """
        plot optical flow
        optical flow have 2 channel : u ,v indicate displacement
    """
    hsv = np.zeros(flow.shape[:2] + (3,)).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.title('optical flow')

def show_result(ret, model_type):
    if  model_type == 'flow':
        plot_velocity_vector(ret)
        flow2color(ret)
    elif model_type == 'stereo':
        plt.imshow(ret)

def strCmp(s):
    return int(s.split('/')[-1].split('_')[0][1:])

def readRawData(file_path, cam_flag):
    cam_list = []
    file_list_name = []
    for dir_name, _, file_list in os.walk(file_path):
        for name in file_list:
            if cam_flag == 'cam2':
                if fnmatch.fnmatch(name, '*cam2*.jpg'):
                    cam_list.append(os.path.join(dir_name, name))
                    file_list_name.append(os.path.join('/mnt/scratch/data/flow_data/', '/'.join(dir_name.split('/')[4:]), '.'.join(name.split('.')[:-1]) + '.npz'))
            else:
                if fnmatch.fnmatch(name, '*cam1*.jpg'):
                    if dir_name.split('/')[-1] == '2016-06-21_1246':
                        cam_list.append(os.path.join(dir_name, name))
                        # file_list_name.append(os.path.join('/mnt/scratch/data/flow_data/', '/'.join(dir_name.split('/')[4:]), '.'.join(name.split('.')[:-1]) + '.npz'))
                        file_list_name.append(os.path.join('/mnt/scratch/dandichen/compressedBenchmark/20160621-2-test', dir_name.split('/')[-1], name[:-4] + '.png'))

    cam_list.sort(key=strCmp)
    file_list_name.sort(key=strCmp)
    return cam_list, file_list_name

global_max = 512
global_min = -512

def main():
    # if len(sys.argv) != 3:
    #     print 'usage:'
    #     print '  python example.py <i mage1 path> <image2 path>'
    #     exit(0)

    file_path = '/mnt/scratch/sync_sd/car_record/demo/20160621-2/binocular_camera'
    file_list, file_list_name = readRawData(file_path, 'cam1')

    for idx in range(len(file_list) - 1):
        img_name1 = file_list[idx].split('/')[-1]
        img_name2 = file_list[idx + 1].split('/')[-1]
        if int(file_list[idx+1].split('/')[-1].split('_')[0][1:]) - int(file_list[idx].split('/')[-1].split('_')[0][1:]) == 1:
            print img_name1, img_name2
            with open(file_list[idx], 'rb') as image_file:
                encoded_image1 = base64.b64encode(image_file.read())

            with open(file_list[idx + 1], 'rb') as image_file:
                encoded_image2 = base64.b64encode(image_file.read())

            payload = {'image1_base64': encoded_image1, 'image1_name': img_name1,
                       'image2_base64': encoded_image2, 'image2_name': img_name2}
            # tic = time.time()
            r = requests.post('http://192.168.1.42:32908/v1/analyzer/opticalflow', json=payload)
            result = json.loads(r.text)
            # toc = time.time()
            # print toc - tic

            shape = result['opticalflow']['shape']
            # model_type = result['opticalflow']['model_type']
            img_stream = StringIO.StringIO(base64.b64decode(result['opticalflow']['base64_data']))
            img = Image.open(img_stream)
            data = np.uint8(np.array(img.getdata()).reshape((shape[0], shape[1], 3)))

            cv2.imwrite(file_list_name[idx], data)

            # data = np.frombuffer(base64.b64decode(result['opticalflow']['base64_data']), dtype=np.float32)
            # data = np.reshape(data, shape)

            # show_result(data, model_type)
            # if not os.path.exists('/'.join(file_list_name[idx].split('/')[:-1])):
            #     os.mkdir('/'.join(file_list_name[idx].split('/')[:-1]))
            # np.savez(file_list_name[idx], data=data)



if __name__ == '__main__':
    main()