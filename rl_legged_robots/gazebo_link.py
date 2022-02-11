import sys
import rospy
from std_srvs.srv import Empty
import gazebo_msgs.msg as gzmsg
import gazebo_msgs.srv as gzsrv

class GazeboLink():
    
    def __init__(self):
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', gzsrv.SetModelState)
    
    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            print("Pausing Simulation...", file=sys.stderr)
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed", file=sys.stderr)
        
    def unpause_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
            print("Unpausing Simulation", file=sys.stderr)
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed", file=sys.stderr)
        
    def reset_sim(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed", file=sys.stderr)

    def reset_world(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed", file=sys.stderr)


    def set_model_state(self, msg):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(msg)
        except rospy.ServiceException as e:
            print('/gazebo/set_model_state service call failed', file=sys.stderr)