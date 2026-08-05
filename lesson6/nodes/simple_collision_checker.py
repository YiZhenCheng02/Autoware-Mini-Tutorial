#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from autoware_mini.msg import Path, DetectedObjectArray, StopLineStatus, StopLineStatusArray
from sensor_msgs.msg import PointCloud2

# Collision point categories: 0 = none, 1 = goal, 2 = traffic light, 3 = static obstacle, 4 = moving obstacle
DTYPE = np.dtype([
    ('x', np.float32),          # position
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),         # velocity
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),   # safety distance before collision point
    ('deceleration_limit', np.float32), # max allowed deceleration (np.inf = no limit)
    ('category', np.int32)
])


class SimpleCollisionChecker:

    def __init__(self):

        # Parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        # add braking_safety_distance_stopline parameter,
        # add lanelet2 map and extract the stop lines with traffic lights
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        
        lanelet2_map = load(lanelet2_map_path, projector)
        self.stop_lines = self.get_traffic_light_stop_lines(lanelet2_map)

        # Variables
        self.detected_objects = None
        self.goal_point = None
        # Add stopline_statuses dict
        self.stopline_statuses = {}

        # Lock for thread safety
        self.lock = threading.Lock()

        # Publishers
        self.collision_points_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=None, tcp_nodelay=True)
        # traffic_light_status subscriber
        rospy.Subscriber('/detection/traffic_light_status', StopLineStatusArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True) 

        rospy.loginfo("%s - initialized", rospy.get_name())

    def traffic_light_status_callback(self, msg):
        with self.lock:
            self.stopline_statuses = {status.stop_line_id: status.status for status in msg.statuses}

    @staticmethod
    def get_traffic_light_stop_lines(lanelet2_map):
        stop_lines = {}
        for reg_el in lanelet2_map.regulatoryElementLayer:
            if reg_el.attributes["subtype"] == "traffic_light":
                stop_line = reg_el.parameters["ref_line"][0]
                stop_lines[stop_line.id] = shapely.LineString([(p.x, p.y) for p in stop_line])
        return stop_lines

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def global_path_callback(self, msg):
        if len(msg.waypoints) > 0:
            self.goal_point = msg.waypoints[-1].position
        else:
            self.goal_point = None

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
            goal_point = self.goal_point

        collision_points = np.array([], dtype=DTYPE)

        if len(msg.waypoints) == 0:
            collision_points_msg = PointCloud2()
            collision_points_msg.header = msg.header
            self.collision_points_pub.publish(collision_points_msg)
            return

        # Create obstacle collision points.
        #         - Create a Shapely LineString from the local path waypoints
        #         - Buffer it with safety_box_width / 2 (cap_style="flat")
        #         - If detected_objects is not None and not empty, iterate over them
        #           and add collision points from their intersections with the buffered path
        #           to the collision_points array
        local_path_xyz = [(waypoint.position.x, waypoint.position.y, waypoint.position.z) for waypoint in msg.waypoints]
        local_path_linestring = shapely.LineString(local_path_xyz)
        local_path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
        shapely.prepare(local_path_buffer)

        if detected_objects is not None and len(detected_objects) > 0:
            for obj in detected_objects:
                object_polygon = shapely.Polygon(np.array(obj.convex_hull).reshape(-1, 3))

                if local_path_buffer.intersects(object_polygon):
                    # Calculate the intersection geometry and create a collision point from each
                    # of its coordinates, filling in the rest of the DTYPE fields from the object
                    # metadata.
                    intersection_points = local_path_buffer.intersection(object_polygon)
                    object_speed = math.sqrt(obj.velocity.x ** 2 + obj.velocity.y ** 2)
                    category = 4 if object_speed > self.stopped_speed_limit else 3

                    for x, y in shapely.get_coordinates(intersection_points):
                        collision_points = np.append(collision_points, np.array([(
                            x, y, obj.centroid.z,
                            obj.velocity.x, obj.velocity.y, obj.velocity.z,
                            self.braking_safety_distance_obstacle,
                            np.inf,
                            category
                        )], dtype=DTYPE))

        # Add goal point as collision point.
        #         - Check if goal_point is within the buffered local path
        #         - If so, append it as a collision point with category=1, zero velocity,
        #           distance_to_stop=braking_safety_distance_goal
        if goal_point is not None:
            goal_point_shapely = shapely.Point(goal_point.x, goal_point.y)
            if local_path_buffer.intersects(goal_point_shapely.buffer(0.1)):
                collision_points = np.append(collision_points, np.array([(
                    goal_point.x, goal_point.y, goal_point.z,
                    0.0, 0.0, 0.0,
                    self.braking_safety_distance_goal,
                    np.inf,
                    1
                )], dtype=DTYPE))

        # add stop line collision points for red traffic lights
        with self.lock:
            current_stopline_statuses = self.stopline_statuses.copy()
            
        for stop_line_id, status in current_stopline_statuses.items():
            # Check if status is RED (STATUS_STOP) and if the stop line exists in our map dictionary
            if status == StopLineStatus.STATUS_STOP and stop_line_id in self.stop_lines:
                stop_line = self.stop_lines[stop_line_id]
                
                if stop_line.intersects(local_path_linestring):
                    intersection = stop_line.intersection(local_path_linestring)
                    
                    for coords in shapely.get_coordinates(intersection):
                        x, y = coords[0], coords[1]
                        
                        collision_points = np.append(collision_points, np.array([(
                            x, y, 0.0,  # z coordinate set to 0
                            0.0, 0.0, 0.0,  # zero velocity
                            self.braking_safety_distance_stopline,
                            np.inf,
                            2  # category 2 for traffic lights
                        )], dtype=DTYPE))

        # Publish the collision points (an empty array means no collision points on the path)
        if len(collision_points) > 0:
            collision_points_msg = msgify(PointCloud2, collision_points)
        else:
            collision_points_msg = PointCloud2()
        collision_points_msg.header = msg.header
        self.collision_points_pub.publish(collision_points_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('simple_collision_checker', log_level=rospy.INFO)
    node = SimpleCollisionChecker()
    node.run()
