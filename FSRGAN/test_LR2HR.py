# -*- coding: utf8 -*-
# ! /usr/bin/python


from FSRGAN import *
import cv2
from PIL import Image

if __name__ == '__main__':

    ## create folders to save result images
    save_dir = "samples/evaluate"
    tl.files.exists_or_mkdir(save_dir)
    # load image
    valid_lr_img = cv2.imread('./data/1.jpg') 

    #valid_lr_img = cv2.resize(valid_lr_img, (32, 32))
    #height = valid_lr_img.shape[0]
    #width = valid_lr_img.shape[1]

    lr= LHFSRGAN()

    out_img,_ = lr.LR_to_HR(valid_lr_img)

    #print(out_img)

    tl.vis.save_image(out_img, save_dir + '/fsran_gen1.png')    
    print("------------- operate image1 done! ------------")

    out_img = lr.LR_to_HR(valid_lr_img)
    tl.vis.save_image(out_img, save_dir + '/fsran_gen2.png')    
    print("------------- operate image2 done! ------------")

    out_img = lr.LR_to_HR(valid_lr_img)
    tl.vis.save_image(out_img, save_dir + '/fsran_gen3.png')    
    print("------------- operate image3 done! ------------")
    
    out_img = lr.LR_to_HR(valid_lr_img)
    tl.vis.save_image(out_img, save_dir + '/fsran_gen4.png')    
    print("------------- operate image3 done! ------------")



