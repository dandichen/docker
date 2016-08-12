import os
import fnmatch
import requests
import base64
import json
import cv2
import time
import numpy as np
import stereovision.calibration as calibration

calib = calibration.StereoCalibration(input_folder='/mnt/scratch/yi/docker_test/depth/calibration_0617')
pixel_means = np.array([103.939, 116.779, 123.68])

def transform(im, pixel_means):
    im = cv2.resize(im, (768, 384))
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] * 0.0039216 - pixel_means[2 - i]
    return im_tensor

def strCmp(s):
    return int(s.split('/')[-1].split('_')[0][1:])

def readRawData(file_path):
    cam1_list = []
    cam2_list = []
    cam1_file_name_list = []

    for dir_name, _, file_list in os.walk(file_path):
        for name in file_list:
            if fnmatch.fnmatch(name, '*cam1*.jpg'):
                cam1_list.append(os.path.join(dir_name, name))
                cam1_file_name_list.append(os.path.join('/mnt/scratch/data/depth_data/', '/'.join(dir_name.split('/')[4:]), '.'.join(name.split('.')[:-1]) + '.npz'))

            if fnmatch.fnmatch(name, '*cam2*.jpg'):
                cam2_list.append(os.path.join(dir_name, name))

    cam1_list.sort(key=strCmp)
    cam2_list.sort(key=strCmp)
    cam1_file_name_list.sort(key=strCmp)
    return cam1_list, cam2_list, cam1_file_name_list

def main():
    # if len(sys.argv) != 3:
    #     print 'usage:'
    #     print '  python example.py <image1 path> <image2 path>'
    #     exit(0)

    file_path = '/mnt/scratch/sync_sd/car_record/demo/20160621-2/binocular_camera'
    cam1_list, cam2_list, cam1_file_name_list = readRawData(file_path)

    f = open('value.txt', 'wa')
    for idx1, idx2 in zip(range(len(cam1_list)), range(len(cam2_list))):
        img_name1 = cam1_list[idx1].split('/')[-1]
        img_name2 = cam2_list[idx2].split('/')[-1]

        if img_name1.split('_')[0][1:] == img_name2.split('_')[0][1:]:
            # print 'idx:', idx1, idx2, 'img name:', img_name1, img_name2
            img1 = cv2.imread(cam1_list[idx1])
            img2 = cv2.imread(cam2_list[idx2])

            img1, img2 = calib.rectify((img1, img2))
            img1 = img1[168:672, 64:960, :]
            img2 = img2[168:672, 64:960, :]
            img1 = transform(img1, np.array([0.411451, 0.432060, 0.450141]))
            img2 = transform(img2, np.array([0.411451, 0.432060, 0.450141]))

            encoded_image1 = base64.b64encode(img1)
            encoded_image2 = base64.b64encode(img2)
            payload = {'image1_base64': encoded_image1, 'image1_shape': img1.shape,
                       'image2_base64': encoded_image2, 'image2_shape': img2.shape}
            # tic = time.time()
            r = requests.post('http://192.168.1.162:32770/v1/analyzer/stereo', json=payload)
            result = json.loads(r.text)
            # toc = time.time()

            shape = result['stereo']['shape']
            data = np.frombuffer(base64.b64decode(result['stereo']['base64_data']), dtype=np.float32)
            data = np.reshape(data, shape)
            print 'idx:', idx1, idx2, 'img name:', img_name1, img_name2, 'max = ', np.amax(data), 'min = ', np.amin(data)
            f.write("%s %s %s %s %s %s %s %s %s %s\n" % ('idx:', str(idx1), str(idx2), 'img name:', img_name1, img_name2, 'max = ', np.amax(data), 'min = ', np.amin(data)))
            # ret = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            # ret2 = cv2.applyColorMap(ret.astype('uint8'), cv2.COLORMAP_JET)
            # ret2 = cv2.resize(ret2, (896, 504))
            # cv2.imshow('2', ret2.astype('uint8'))
            # cv2.waitKey()
            # if not os.path.exists('/'.join(cam1_file_name_list[idx1].split('/')[:-1])):
            #     os.mkdir('/'.join(cam1_file_name_list[idx1].split('/')[:-1]))
            # np.savez(cam1_file_name_list[idx1], data=data)

        else:
            raise ValueError('cam1 and cam2 are nor synchronized')
    f.close()

# if __name__ == '__main__':
#     main()