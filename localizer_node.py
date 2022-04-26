import random
import cv2
import socket
from ctypes import *
import numpy as np
import binascii
import rospy
import tf
import time
from geometry_msgs.msg import Pose2D
from scipy.spatial.transform import Rotation

class AprilTag:
    def __init__(self, id, center, quat):
        self.id = id
        self.center = center
        self.quat = quat
        self.angle = 0.0

    def getWPose2D(self,wR,wT):
        Rtc = Rotation.from_quat(self.quat).as_dcm()
        Rcw = wR
        Rtw = np.dot(Rcw,Rtc)
        dw = np.dot(Rtw, np.array([[0.0],[-1.0],[0.0]]))
        self.angle = -np.arctan2(-dw[1,0],dw[0,0])*180.0/np.pi
        wc = np.dot(Rcw, self.center.reshape((3,1))) + wT.reshape((3,1))
        return wc[0,0], wc[1,0], self.angle

        



# Positions (multiples of 4 bytes) , and Length in bytes
p_time = (4, 8)

# Without header positions
p_id = (0, 4)
p_ncode = (2, 4)
p_c0 = (3, 4)
p_c1 = (4, 4)
p_p10 = (5, 4)
p_p11 = (6, 4)
p_p20 = (7, 4)
p_p21 = (8, 4)
p_p30 = (9, 4)
p_p31 = (10, 4)
p_p40 = (11, 4)
p_p41 = (12, 4)

# Global Variables
g_D = [0.0902, 0.1746, 0.0, 0.0, -1.1719]
g_K = np.array([[1739.3961, 0.0, 950.8745],
                [0.0, 1740.8267, 545.3859],
                [0.0, 0.0, 1.0]])
g_s = 0.47002208
g_R = np.eye(3)
g_t = np.zeros((3,))

# Tag to 3D associations
# g_associations = {3: (0,0,0.025),
#                   2: (1.2,0.0,0.125),
#                   4: (1.2,0.6,0.06),
#                   1: (0.6,0.6,0.1)}


g_associations = {4: np.array([  -1.757, 1.22, 0.165  ]),
                  2: np.array([  0.90, 0.863, 0.083  ]),
                  5: np.array([  0.914, -1.23, 0.265  ]),
                  1: np.array([  -1.239, -0.203, 0.11  ])}

# ud = 0.6
# g_associations = {4: np.array([  3.0,-2.0, 0.169/ud  ])*ud,
#                   2: np.array([  0.0,0.0, 0.189/ud  ])*ud,
#                   5: np.array([  3.0,-1.0, 0.217/ud  ])*ud,
#                   1: np.array([  2.0,0.0, 0.185/ud  ])*ud}
g_square_half_lengths = {4: 0.168/2.0,
                         2: 0.168/2.0,
                         5: 0.168/2.0,
                         1: 0.168/2.0,
                         3: 0.168/2.0}

g_pub = rospy.Publisher('atagpose', Pose2D, queue_size=1)


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape
    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)
    C = np.dot(np.transpose(centeredP), centeredQ) / n
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    R = np.dot(V, W)
    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor
    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)
    return c, R.transpose(), t


def h2f(s):
    i = int(s, 16)                   # convert from hex to a Python int
    cp = pointer(c_int(i))           # make this into a c integer
    fp = cast(cp, POINTER(c_float))  # cast the int pointer to a float pointer
    return fp.contents.value 

def read_part(main_string, position):
    return main_string[position[0]*2*4 : position[0]*2*4+position[1]*2]

def get3Dcoordinates(D,K,s,R,t,x,y):
    xpp = (x-K[0,2])/K[0,0]
    ypp = (y-K[1,2])/K[1,1]
    r = np.sqrt(xpp**2+ypp**2)
    xc = x*(1.0+D[0]*r**2+D[1]*r**4+D[4]*r**6)
    yc = y*(1.0+D[0]*r**2+D[1]*r**4+D[4]*r**6)
    X = s*(xc-K[0,2])/K[0,0]
    Y = s*(yc-K[1,2])/K[1,1]
    Z = s*1.0
    return np.array([X,Y,Z])

def transformNscale3D(s, R, t, coord):
    return (np.matmul(R, (s*coord.transpose()))).transpose() + t

def get3DforMarker(marker_id, p0, p1, p2, p3, c):
    model_points = [
                        [ -g_square_half_lengths[marker_id],  g_square_half_lengths[marker_id], 0 ],
                        [  g_square_half_lengths[marker_id],  g_square_half_lengths[marker_id], 0 ],
                        [  g_square_half_lengths[marker_id], -g_square_half_lengths[marker_id], 0 ],
                        [ -g_square_half_lengths[marker_id], -g_square_half_lengths[marker_id], 0 ],
                        [ 0.0, 0.0, 0 ]
                   ]
    image_points = [p0,p1,p2,p3,c]
    (_, rvec, tvec) = cv2.solvePnP(np.array(model_points), np.array(image_points), g_K, np.array(g_D))
    rtheta = np.linalg.norm(rvec)
    rdir = (rvec/rtheta)*np.sin(rtheta/2.0)
    rquat = (rdir[0,0], rdir[1,0], rdir[2,0], np.cos(rtheta/2.0))
    br = tf.TransformBroadcaster()
    br.sendTransform((tvec[0,0], tvec[1,0], tvec[2,0]),
                     rquat,
                     rospy.Time.now(),
                     'tag:'+str(marker_id),
                     "camera")
    return rvec[:,0], tvec[:,0], rquat


def process_apriltag_msg(hex_string):
    time = int(read_part(hex_string, p_time), 16)
    atags = []

    if(len(hex_string)>48): # April Tag Detected

        without_header = hex_string[48:]
        #print "Tags detected:", len(without_header)/(2*88)
        for i in range(int(len(without_header)/(2*88))):
            tag_string = without_header[i*88*2:(i+1)*88*2]
            tag_id = int(read_part(tag_string, p_id), 16)
            ncode = int(read_part(tag_string, p_ncode), 16)
            c0 = h2f(read_part(tag_string, p_c0))
            c1 = h2f(read_part(tag_string, p_c1))
            p10 = h2f(read_part(tag_string, p_p10))
            p11 = h2f(read_part(tag_string, p_p11)) 
            p20 = h2f(read_part(tag_string, p_p20))
            p21 = h2f(read_part(tag_string, p_p21))
            p30 = h2f(read_part(tag_string, p_p30))
            p31 = h2f(read_part(tag_string, p_p31))
            p40 = h2f(read_part(tag_string, p_p40))
            p41 = h2f(read_part(tag_string, p_p41))

            rvec, tvec, quat = get3DforMarker(tag_id, [p10,p11], [p20,p21], [p30,p31], [p40,p41], [c0,c1])
            atags.append(AprilTag(tag_id,tvec,quat))
    return atags

def calibrate_AprilTags(atags):
    global g_s, g_R, g_t
    lP = []
    lQ = []
    for atag in atags:
        if atag.id in g_associations:
            lP.append(atag.center)
            lQ.append(g_associations[atag.id])
    P = np.array(lP)
    Q = np.array(lQ)
    if P.shape[0] >=3 :
        c,R,t = umeyama(P,Q)
        g_s = c
        g_R = c*R
        g_t = t
        q = Rotation.from_dcm(c*R).as_quat()
        br = tf.TransformBroadcaster()
        br.sendTransform((t[0], t[1], t[2]),
                        (q[0], q[1], q[2], q[3]),
                        rospy.Time.now(),
                        'camera', # From 
                        "world")  # To

def localize_AprilTags(atags):
    for atag in atags:
        if atag.id not in g_associations and atag.id==3:
            x,y,theta = atag.getWPose2D(g_R, g_t)
            msg = Pose2D(x,y,theta)
            g_pub.publish(msg)
            # print x,y,theta



def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('192.168.10.119', 7709))
    rospy.init_node('ataglocalizer', anonymous=True)
    rate = rospy.Rate(2)
    prev_time = time.time()
    while not rospy.is_shutdown():
        curr_time = time.time()
        message, address = server_socket.recvfrom(5*112)
        if(curr_time-prev_time > 1./2):
            byte_array = bytearray(message)
            hexadecimal_string = binascii.hexlify(byte_array)
            atags = process_apriltag_msg(hexadecimal_string)
            calibrate_AprilTags(atags)
            localize_AprilTags(atags)
            prev_time = time.time()

main()

    
