import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
import time

g_target = (1.2,-0.6,45 * np.pi/180.0)
g_aspeed = 0.2
g_delt_threshold = 2.5*np.pi/180.0
g_delp_threshold = 0.05
g_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

g_K_theta = 2.0
g_K_x = 0.1

def callback(data):
    global g_target, g_K_theta
    gx,gy,gt = g_target
    ## First am I oriented correctly?
    x,y,t = data.x, data.y, data.theta*np.pi/180.0
    dirv = np.array([np.cos(t), np.sin(t)])
    tdirv = np.array([gx-x, gy-y])
    print tdirv, t
    delt1 = np.arctan2(dirv[0]*tdirv[1]-dirv[1]*tdirv[0],dirv[0]*tdirv[0]+dirv[1]*tdirv[1])
    delt2 = np.arctan2(dirv[0]*(-tdirv[1])-dirv[1]*(-tdirv[0]),dirv[0]*(-tdirv[0])+dirv[1]*(-tdirv[1]))
    delt = 0.0
    multp = 1.0
    if abs(delt1) < abs(delt2):
        delt = delt1
    else:
        multp = -1.0
        delt = delt2
    delp = np.linalg.norm(tdirv)
    print(delt1, delt2, delt, ' | ',  delp, g_delt_threshold)
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    # if np.abs(delt) > g_delt_threshold:
    cmd.angular.z = np.clip(g_K_theta*delt, -0.6, 0.6)
    # elif delp > g_delp_threshold:
    cmd.linear.x = np.clip(multp*g_K_x*delp, -0.5, 0.5)
    g_pub.publish(cmd)
    rospy.sleep(2.5)


    
def listener():
    rospy.init_node('controller', anonymous=True)
    rospy.Subscriber("atagpose", Pose2D, callback, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    listener()
