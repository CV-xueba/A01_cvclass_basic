# Author: Ji Qiu （BUPT）
# cv_xueba@163.com

import cv2
import numpy as np


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print ('Get_gradient_img')
        
        new_img_x = np.zeros([self.y, self.x], dtype=np.float)
        new_img_y = np.zeros([self.y, self.x], dtype=np.float)
        for i in range(0, self.x):
            for j in range(0, self.y):
                if j == 0:
                    new_img_y[j][i] = 1
                else:
                    new_img_y[j][i] = np.sum(np.array([[self.img[j - 1][i]], [self.img[j][i]]]) * self.y_kernal)
                if i == 0:
                    new_img_x[j][i] = 1
                else:
                    new_img_x[j][i] = np.sum(np.array([self.img[j][i - 1], self.img[j][i]]) * self.x_kernal)

        gradient_img, self.angle = cv2.cartToPolar(new_img_x, new_img_y)
        self.angle = np.tan(self.angle)
        self.img = gradient_img.astype(np.uint8)
        return self.img

    def Non_maximum_suppression (self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print ('Non_maximum_suppression')
        
        result = np.zeros([self.y, self.x])
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if abs(self.img[i][j]) <= 4:
                    result[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    # g1 g2
                    #    C
                    #    g4 g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #    g2 g1
                    #    C
                    # g3 g4
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                else:
                    gradient2 = self.img[i][j - 1]
                    gradient4 = self.img[i][j + 1]
                    # g1
                    # g2 C g4
                    #      g3
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    #      g3
                    # g2 C g4
                    # g1
                    else:
                        gradient3 = self.img[i - 1][j + 1]
                        gradient1 = self.img[i + 1][j - 1]

                temp1 = abs(self.angle[i][j]) * gradient1 + (1 - abs(self.angle[i][j])) * gradient2
                temp2 = abs(self.angle[i][j]) * gradient3 + (1 - abs(self.angle[i][j])) * gradient4
                if self.img[i][j] >= temp1 and self.img[i][j] >= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
        self.img = result
        return self.img

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print ('Hysteresis_thresholding')
        
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
                    if abs(self.angle[i][j]) < 1:
                        if self.img_origin[i - 1][j] > self.HT_low_threshold:
                            self.img[i - 1][j] = self.HT_high_threshold
                        if self.img_origin[i + 1][j] > self.HT_low_threshold:
                            self.img[i + 1][j] = self.HT_high_threshold
                        # g1 g2
                        #    C
                        #    g4 g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #    g2 g1
                        #    C
                        # g3 g4
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i - 1][j + 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                    else:
                        if self.img_origin[i][j - 1] > self.HT_low_threshold:
                            self.img[i][j - 1] = self.HT_high_threshold
                        if self.img_origin[i][j + 1] > self.HT_low_threshold:
                            self.img[i][j + 1] = self.HT_high_threshold
                        # g1
                        # g2 C g4
                        #      g3
                        if self.angle[i][j] < 0:
                            if self.img_origin[i - 1][j - 1] > self.HT_low_threshold:
                                self.img[i - 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j + 1] = self.HT_high_threshold
                        #      g3
                        # g2 C g4
                        # g1
                        else:
                            if self.img_origin[i - 1][j + 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
                            if self.img_origin[i + 1][j - 1] > self.HT_low_threshold:
                                self.img[i + 1][j - 1] = self.HT_high_threshold
        return self.img

    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()
        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img
