import numpy as np
import cv2
import skimage.color as color
import skimage.io as io
import pylab
import os

from glob import glob

begin = 1
end = 100
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
im_Dir = '/media/li/Li/casia/CASIA1/Sp'
mask_Dir = '/media/li/Li/casia/CASIA1/mask'
auth_Dir = '/media/li/Li/casia/CASIA1/Au'
ext = 'Sp*'
dataType = '.jpg'
filenames = glob(os.path.join(im_Dir, ext))
im_num = len(filenames)


im_id = 0
for im in filenames:

    im_id += 1
    print(' {:d}/{:d} images'.format(im_id, im_num))
    img_org = io.imread(im)
    img = img_org.copy()
    img_org = cv2.cvtColor(img_org, cv2.COLOR_RGB2BGR)
    base_name = os.path.splitext(os.path.basename(im))[0]
    content = base_name.split("_")
    au_tmp = str(content[4])
    auth_name = 'Au_' + au_tmp[:-4] + '_' + au_tmp[-4:] + '*'

    auth_img = io.imread(glob(os.path.join(auth_Dir, auth_name))[0])
    if (img.shape != auth_img.shape):
        au_tmp = str(content[5])
        auth_name = 'Au_' + au_tmp[:-4] + '_' + au_tmp[-4:] + '*'
        auth_img = io.imread(glob(os.path.join(auth_Dir, auth_name))[0])
    img = color.rgb2grey(img)
    auth_img = color.rgb2grey(auth_img)

    mask = np.abs(img - auth_img) * 255

    ret, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)#After testing, the threshold is set to 15
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)#Remove noise
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)#Binarization

    # Generate a new mask that contains only large regions
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda i: len(i), reverse=True)
    mask_save = np.zeros_like(mask, dtype=np.uint8)
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        if (area > (mask.shape[0] / 20 * mask.shape[1] / 20) and area < (mask.shape[0] * mask.shape[1] * 0.7)):
            c_max.append(cnt)
    cv2.drawContours(mask_save, c_max, -1, (255), thickness=-1)
    if len(c_max) > 0:
        cv2.imwrite('/media/li/Li/casia/CASIA1/probe/' + str(base_name) + ".png", img_org)
        cv2.imwrite('/media/li/Li/casia/CASIA1/mask/' + str(base_name) + ".png", mask_save)
    else:
        continue
