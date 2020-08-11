import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def compute_F(points_1, points_2):
    N=points_1.shape[0]
    m_1=np.average(points_1,axis=0)
    m_2=np.average(points_2,axis=0)
    mean_1=points_1-m_1.reshape(1,2)
    mean_2=points_2-m_2.reshape(1,2)
    p1_sum=np.sum((mean_1)**2,axis=None)
    p2_sum=np.sum((mean_2)**2,axis=None)
    p1=(p1_sum/(2*N))**0.5
    p1_inv=1/p1
    p2=(p2_sum/(2*N))**0.5
    p2_inv=1/p2
    x=mean_1*p1_inv
    y=mean_2*p2_inv
    A = np.zeros((N,9))
    for i in range(N):
        A[i] = [x[i,0]*y[i,0], x[i,1]*y[i,0], y[i,0],x[i,0]*y[i,1],
               x[i,1]*y[i,1], y[i,1],x[i,0], x[i,1], 1]
    u,s,vt = np.linalg.svd(A,full_matrices=True)
    F = vt[8,:].reshape(3,3)
    U, S, Vt = np.linalg.svd(F, full_matrices=True)
    S[2] = 0
    Smat = np.diag(S)
    F = np.dot(U, np.dot( Smat, Vt))
    T_1 = np.zeros((3,3))
    T_1[0,0] = p1_inv
    T_1[1,1] = p1_inv
    T_1[2,2] = 1
    T_1[0,2] = -p1_inv*m_1[0]
    T_1[1,2] = -p1_inv*m_1[1]

    T_2 = np.zeros((3,3))
    T_2[0,0] = p2_inv
    T_2[1,1] = p2_inv
    T_2[2,2] = 1
    T_2[0,2] = -p2_inv*m_2[0]
    T_2[1,2] = -p2_inv*m_2[1]
    F = np.dot( np.transpose(T_2), np.dot( F, T_1 ))

    return F

def sampson_distance(x1, x2, F):

    x1 = np.vstack((x1.T,np.ones(x1.T.shape[1])))
    x2 = np.vstack((x2.T,np.ones(x2.T.shape[1])))

    num = (np.linalg.det(np.dot(x1.T,np.dot(F,x2))))**2

    Fx1 = np.dot(F,x1)
    Fx2 = np.dot(F,x2)
    denominator = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2

    #denominator = (np.dot(F,x1.T)[0])**2 + (np.dot(F,x2.T)[0])**2 + (np.dot(F,x1.T)[1])**2 + (np.dot(F,x2.T)[1])**2
    sampson = num/denominator

    return sampson

def ransac(x1, x2):
    points_idx = []

    # just to have an high starting point
    temp = 1000000000
    for i in range(x1.shape[0]):
        points_idx = random.sample(range(x1.shape[0]),8)
        F = compute_F(x1[points_idx],x2[points_idx])

        d = sampson_distance(x1[points_idx],x2[points_idx], F)
        print(np.sum(d))
        if np.sum(d)<temp:
            print('better')
            best_F = F
            temp = np.sum(d)

    return best_F

def draw_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    m,n = im.shape[:2]
    line = np.dot(F,x)
    # epipolar line parameter and values
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx],lt[ndx],linewidth=2)
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

def calculate_epipole_point(F):
    # calculates right epipole, use F.T for left epipole
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]
