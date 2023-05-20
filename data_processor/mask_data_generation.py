"""
@author: Yuxi Liu
@date: 20230401
@contact: liuyuxi.tongji@gmail.com
"""

import os
import random
import cv2
# import torch
import numpy as np
# from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
'''
landmarks = np.array([[36.2946, 51.5014],
                      [76.5318, 51.5014],
                      [56.0252, 71.7366],
                      [41.5493, 92.2041],
                      [70.7299, 92.2041]
                      ], dtype=np.float32 )
'''
'''
landmarks = np.array([[30.2946, 51.6963],
                      [65.5318, 51.5014],
                      [48.0252, 71.7366],
                      [33.5493, 92.3655],
                      [62.7299, 92.2041]
                      ], dtype=np.float32 )
'''
# for CALFW-glasses
landmarks0 = np.array([[45.0, 44.0+5],
                      [67.0, 44.0+5],
                      [56.0, 62.0+5],
                      [41.0, 78.0+5],
                      [71.0, 78.0+5]
                      ], dtype=np.float32 )


class MaskData(object):
    def __init__(self, data_root, train_file, landmark_root=None):
        self.data_root = data_root
        self.data_save_root = data_root+'_'
        self.train_list = []
        self.landmark_root = landmark_root
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path=line.split(' ')[0]
            self.train_list.append(image_path)
            line = train_file_buf.readline().strip()
        self.mask_img = cv2.imread("mask_img.png", cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread("glass_img3.png", cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread("sunglass_img2.png", cv2.IMREAD_UNCHANGED)
    def __len__(self):
        return len(self.train_list)
    def __maskitem__(self, index, mask_type):
        image_path = self.train_list[index]
        image_save_path = os.path.join(self.data_save_root, image_path)        
        image_path = os.path.join(self.data_root, image_path)
        # print(self.data_save_root,image_save_path)
        folder_path = os.path.dirname(image_save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image = cv2.imread(image_path)
        # image[25:225, 25:225]
        # image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        if (mask_type=='mask'):
            masked_sample=self.mask_images(sample,landmarks=self.get_landmark(image_path))
            # plt.imshow(masked_sample.astype('uint8'))
            # 将numpy数组转换为PIL Image对象
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save(image_save_path)
        elif (mask_type=='glasses'):
            # print(image_save_path)
            masked_sample=self.mask_images_glass(sample,landmarks=self.get_landmark(image_path))
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            image = cv2.resize(image, (112, 112))
            # 保存PIL Image对象为图片
            # print(image_save_path)
            img.save(image_save_path)
        elif (mask_type=='sunglasses'):
            # print(image_save_path)
            masked_sample=self.mask_images_sunglass(sample,landmarks=self.get_landmark(image_path))
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            image = cv2.resize(image, (112, 112))
            # 保存PIL Image对象为图片
            # print(image_save_path)
            img.save(image_save_path)
    def __maskitem_prob__(self, index):
        image_path = self.train_list[index]
        image_save_path = os.path.join(self.data_save_root, image_path)        
        image_path = os.path.join(self.data_root, image_path)
        # print(self.data_save_root,image_save_path)
        folder_path = os.path.dirname(image_save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image = cv2.imread(image_path)
        # image[25:225, 25:225]
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        prob=random.uniform(0, 3)
        if(prob<=1):
            mask_type='mask'
        elif (prob<=2):
            mask_type='glasses'
        else:
            mask_type='sunglasses'
        if (mask_type=='mask'):
            masked_sample=self.mask_images(sample,landmarks=self.get_landmark(image_path))
            # plt.imshow(masked_sample.astype('uint8'))
            # 将numpy数组转换为PIL Image对象
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save(image_save_path)
        elif (mask_type=='glasses'):
            # print(image_save_path)
            masked_sample=self.mask_images_glass(sample,landmarks=self.get_landmark(image_path))
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # image = cv2.resize(image, (112, 112))
            # 保存PIL Image对象为图片
            # print(image_save_path)
            img.save(image_save_path)
        elif (mask_type=='sunglasses'):
            # print(image_save_path)
            masked_sample=self.mask_images_sunglass(sample,landmarks=self.get_landmark(image_path))
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # image = cv2.resize(image, (112, 112))
            # 保存PIL Image对象为图片
            # print(image_save_path)
            img.save(image_save_path)
    def generate(self, mask_type=None):
        if(mask_type==None):
            self.data_save_root = self.data_root+'_random_mask'
            if not os.path.exists(self.data_save_root):
                os.makedirs(self.data_save_root)
            print(self.data_save_root)
            for index in range(self.__len__()):
                self.__maskitem_prob__(index)
                #if index==100:
                #    return
        else:
            self.data_save_root = self.data_root+'_random_'+mask_type+'2'
            if not os.path.exists(self.data_save_root):
                os.makedirs(self.data_save_root)
            print(self.data_save_root)
            for index in range(self.__len__()):
                self.__maskitem__(index, mask_type)
                #print(index)
                #if index==10:
                #    return
    def get_landmark(self, file_path):
        if self.landmark_root==None:
            return landmarks0
        filename=os.path.basename(file_path)
        filename=filename[:-4]+'_5loc_attri.txt'
        file_path_new = os.path.join(self.landmark_root,filename)
        train_file_buf = open(file_path_new)
        line = train_file_buf.readline().strip()
        landmarks=[]
        while line:
            landmarks.append(line.split(' '))
            landmarks[-1][0]=float(landmarks[-1][0])
            landmarks[-1][1]=float(landmarks[-1][1])
            line = train_file_buf.readline().strip()
        # print(landmarks)
        return landmarks
    def mask_images(self,img,landmarks):
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        '''
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        '''
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1-20*4]), 
                            np.array([580+rx,614+rs+20*1-20*4]), 
                            np.array([1047+rx,614+rs+20*1-20*4]), 
                            np.array([967+rx+60,150+rs-20*5+20*4]), 
                            np.array([660+rx-60,150+rs-20*5+20*4])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_glass(self,img,landmarks):
        # print(landmarks)
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-80,80)
        rx = np.random.randint(-20,20)
        # rs = 0
        # rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1+20*2]), 
                            np.array([580+rx,614+rs+20*1+20*2]), 
                            np.array([1047+rx,614+rs+20*1+20*2]), 
                            np.array([967+rx+60,150+rs-20*5+20*2]), 
                            np.array([660+rx-60,150+rs-20*5+20*2])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_sunglass(self,img,landmarks):
        # print(landmarks)
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-80,80)
        rx = np.random.randint(-20,20)
        # rs = 0
        # rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1+20*2]), 
                            np.array([580+rx,614+rs+20*1+20*2]), 
                            np.array([1047+rx,614+rs+20*1+20*2]), 
                            np.array([967+rx+60,150+rs-20*5+20*2]), 
                            np.array([660+rx-60,150+rs-20*5+20*2])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.sunglass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    cropped_face_folder= "/CIS20/lyx/FaceX-Zoo-main-new/test_data/CALFW_reid"
    image_list_file_path= '/CIS20/lyx/FaceX-Zoo-main-new/test_data/CALFW_train_list.txt'
    landmarks_root="/CIS20/lyx/FaceX-Zoo-main-new/test_data/CA_landmarks/"
    '''
    cropped_face_folder= "/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace"
    image_list_file_path= '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/webface_train_list.txt'
    landmarks_root=None
    '''
    MD=MaskData(cropped_face_folder,image_list_file_path,landmark_root=landmarks_root)
    print('begin!')
    MD.generate('sunglasses')
    print('end!')