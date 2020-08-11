import numpy as np
import cv2
from scipy import ndimage, interpolate


def track(I, J, input_points, total_points, window=(21, 21), min_disp=0.01):

    output = []
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J_gray = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)

    #normalization
    I_norm = I_gray/I_gray.max()
    J_norm = J_gray/J_gray.max()

    for points in input_points:
        print('inside calculate')
        d = calculate_new_point(I_norm, J_norm, points[0], points[1], window)
        if d is not None:
            print('output '+str(output))
            output.append((points[0] + d[0], points[1] + d[1]))
    output = np.asarray(output).T
    output = output.astype(int)
    frame = J.copy()
    for point in zip(*total_points[::-1]):
        print('printing new points')
        print(point)
        print(type(point))
        J = cv2.circle(J, point, 3, (0, 0, 255), 10)

    for point in zip(*output[::-1]):
        print('printing new points')
        print(point)
        print(type(point))
        J = cv2.circle(J, point, 3, (0, 0, 255), 10)
    # for point in zip(*output[::-1]):
    #     frame = cv2.circle(frame, point, 3, (255, 0, 0), 4)

    return J, output

def calculate_new_point(I, J, x, y, window):
    displ_tot = np.array([0., 0.]).T

    # The window to evaluate
    win_x = np.arange(x, x + window[0], dtype=float)
    win_y = np.arange(y, y + window[1], dtype=float)

    roi = I[x:x + window[0], y: y + window[1]]

    # Find image gradient in I
    Ix = cv2.Sobel(roi,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(roi,cv2.CV_64F,0,1,ksize=3)

    # Calculate the Hessian matrix
    Ix = Ix.flatten()
    Iy = Iy.flatten()
    A = np.array([Ix, Iy])
    T = A.dot(A.T)
    #T = np.matmul(A, A.T)
    # Check that H is not singular
    if np.linalg.det(T) == 0:
        return None
    T_inv = np.linalg.inv(T)

    # Bilinear interpolation
    x_arr = np.arange(0, J.shape[1])
    y_arr = np.arange(0, J.shape[0])
    J_bilinear = interpolate.interp2d(x_arr, y_arr, J, kind='linear')


    for x in range(35):
        try:
            # Calculate e matrix
            J_window = J_bilinear(win_x + displ_tot[0], win_y + displ_tot[1])
            D = (I[x:x + window[0], y: y + window[1]]-J_window).flatten()
            e = -1*(np.dot(A,D))
            d_temp = np.dot(T_inv, e)
            displ_tot = displ_tot + d_temp

            return displ_tot
        except:
            return None

    # calculate displacement


def compute_corners(img, threshold=0.5):

    img_cpy = img.copy()

    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Ix = Convolution.convolution(img_gray, 'sobel_x')
    #Iy = Convolution.convolution(img_gray, 'sobel_y')
    Ix = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixdy = Ix*Iy

    #g_Ix2 = Convolution.convolution(dx2, 'gaussian')
    #g_Iy2 = Convolution.convolution(dy2, 'gaussian')
    #g_IxIy = Convolution.convolution(dxdy, 'gaussian')

    g_Ix2 = cv2.GaussianBlur(Ix2, (3,3),0)
    g_Iy2 = cv2.GaussianBlur(Iy2, (3,3),0)
    g_IxIy = cv2.GaussianBlur(Ixdy, (3,3),0)

    R = g_Ix2*g_Iy2 - np.square(g_IxIy) - 0.22*np.square(g_Ix2 + g_Iy2)

    # find all points above threshold
    img_cpy[R>threshold]=[255,0,0]

    return img_cpy, np.where(R > threshold*R.max())

cap = cv2.VideoCapture('Ass_img/MarinaBayatNightDrone.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Capture frame-by-frame
ret, frame = cap.read()
cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 900,600)

old_frame, points = compute_corners(frame)

points = np.asarray(points)
total_points = points
print(len(points.T))
cv2.imshow('Frame',old_frame)

# Read until video is completed
while(cap.isOpened()):
    ret, new_frame = cap.read()
    old_frame, points = track(old_frame, new_frame, points.T, total_points)
    cv2.imshow('Frame',old_frame)
    print('points and total points')
    print(points)
    print('total points')
    print(total_points)
    total_points = np.hstack((total_points, points))



    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
#
# corner_det = Corner_Detector()
# corners = corner_det.compute_corners()
