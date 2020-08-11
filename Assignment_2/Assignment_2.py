import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

import ransac


if __name__ == '__main__':

    # create two windows with two images to select the points
    # after selecting a point press "a" to register it
    # in the array, select a point in the first image, then
    # the respective point in the other image
    ix,iy = -1,-1
    # mouse callback function
    def draw_circle1(event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img1,(x,y),10,(0,0,255),-1)
            print(str(x)+' '+str(y))
            points1.append((x,y))
    def draw_circle2(event,x,y,flags,param):
        global ix,iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img2,(x,y),10,(0,255,0),-1)
            print(str(x)+' '+str(y))
            points2.append((x,y))

            # path
    path1 = 'rigidSfM/DSCN0953.JPG'
    path2 = 'rigidSfM/DSCN0954.JPG'

    # Using cv2.imread() method
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)


    img_horizontal = np.hstack((img1, img2))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1600,1600)
    cv2.setMouseCallback('image',draw_circle1)

    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image2', 1600,1600)
    cv2.setMouseCallback('image2',draw_circle2)

    points1 = []
    points2 = []

    x = 0

    while(1):
        cv2.imshow('image',img1)
        cv2.imshow('image2',img2)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        x+=1

    cv2.destroyAllWindows()
    # compute F and then

    p_1 = np.asanyarray(points1)
    p_2 = np.asanyarray(points2)
    # # np.save('p_1', p_1)
    # # np.save('p_2', p_2)
    #
    # p_1 = np.load('p_1.npy')
    # p_2 = np.load('p_2.npy')

    F = ransac.ransac(p_1,p_2)
    e_r = ransac.calculate_epipole_point(F)
    e_l = ransac.calculate_epipole_point(F.T)

    p1 = np.vstack((p_1.T,np.ones(p_1.T.shape[1])))
    p2 = np.vstack((p_2.T,np.ones(p_2.T.shape[1])))

    plt.subplot(121)
    plt.imshow(img1)

    # plot each line individually
    for i in range(9):
        ransac.draw_epipolar_line(img1,F,p2[:,i],e_r,False)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img2)
    for i in range(9):
        ransac.draw_epipolar_line(img2,F.T,p1[:,i],e_l,False)

    plt.axis('off')
    plt.show()
