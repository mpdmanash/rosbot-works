import random
import socket
from ctypes import *
import numpy as np
import binascii
import rospy
from geometry_msgs.msg import Pose2D

class AprilTag:
    def __init__(self, id, center, p1, p2, p3, p4):
        self.id = id
        self.center = center
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.angle = 0.0

    def computeAngle(self):
        D = self.p3-self.p2
        self.angle = -np.arctan2(-D[1],D[0])*180.0/np.pi
        #self.angle = np.arccos(D[0]/np.linalg.norm(D))*180.0/np.pi



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

ud = 0.6
g_associations = {4: np.array([  0.0,0.0, 0.169/ud  ])*ud,
                  2: np.array([  2.0,0.0, 0.189/ud  ])*ud,
                  5: np.array([  2.0,-2.0, 0.217/ud  ])*ud,
                  1: np.array([  1.0,-2.0, 0.185/ud  ])*ud}

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
    return c, R, t


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

            p13d = get3Dcoordinates(g_D,g_K,1.0,None,None,p10,p11)
            p23d = get3Dcoordinates(g_D,g_K,1.0,None,None,p20,p21)
            p33d = get3Dcoordinates(g_D,g_K,1.0,None,None,p30,p31)
            p43d = get3Dcoordinates(g_D,g_K,1.0,None,None,p40,p41)

            c3d = get3Dcoordinates(g_D,g_K,1.0,None,None,c0,c1)
            # a1 = np.linalg.norm(p23d-p13d)
            # a2 = np.linalg.norm(p33d-p23d)
            # a3 = np.linalg.norm(p43d-p33d)
            # a4 = np.linalg.norm(p13d-p43d)

            atags.append(AprilTag(tag_id,c3d,p13d,p23d,p33d,p43d))
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
        g_R = R
        g_t = t

def localize_AprilTags(atags):
    for atag in atags:
        if atag.id not in g_associations and atag.id==3:
            atag.center = transformNscale3D(g_s,g_R,g_t,atag.center)
            atag.p1 = transformNscale3D(g_s,g_R,g_t,atag.p1)
            atag.p2 = transformNscale3D(g_s,g_R,g_t,atag.p2)
            atag.p3 = transformNscale3D(g_s,g_R,g_t,atag.p3)
            atag.p4 = transformNscale3D(g_s,g_R,g_t,atag.p4)
            atag.computeAngle()
            msg = Pose2D(atag.center[0], atag.center[1], atag.angle)
            g_pub.publish(msg)
            #print atag.angle, atag.center 



def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(('192.168.10.119', 7709))
    rospy.init_node('ataglocalizer', anonymous=True)
    while True:
        message, address = server_socket.recvfrom(5*112)
        byte_array = bytearray(message)
        hexadecimal_string = binascii.hexlify(byte_array)
	# print(len(hexadecimal_string))
        atags = process_apriltag_msg(hexadecimal_string)
        calibrate_AprilTags(atags)
        localize_AprilTags(atags)


main()

    
