#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)


class ClusterDetector:
    def __init__(self):
        # Parameters
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # Publishers
        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z', 'label']], dtype=np.float32)

        if msg.header.frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp,
                                                            rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return

            tf_matrix = numpify(transform.transform).astype(np.float32)
            points_copy = points.copy()
            points_copy[:, 3] = 1
            points_copy = points_copy.dot(tf_matrix.T)
            points[:, :3] = points_copy[:, :3]

        # TODO 3: Create a DetectedObjectArray and iterate over cluster labels.
        #         - Create DetectedObjectArray with header (stamp from msg, frame_id from self.output_frame)
        #         - Loop over cluster labels and assign correct points to each cluster
        #         - Skip clusters with fewer points than self.min_cluster_size
        result_object_array = DetectedObjectArray()
        result_object_array.header.stamp = msg.header.stamp
        result_object_array.header.frame_id = self.output_frame

        for i in range(int(max(points[:, 3]) + 1)):
            # Find all points with a correct label
            points3d = points[points[:, 3] == i]

            # Skip clusters with fewer points than self.min_cluster_size
            if len(points3d) < self.min_cluster_size:
                continue

            # TODO 4: Calculate centroid and convex hull for each cluster.
            #         - Centroid: mean of x, y, z coordinates
            #         - Convex hull: use MultiPoint(points3d[:, :2]).convex_hull,
            #           skip clusters whose hull is not a Polygon (line or point)
            #         - Create a DetectedObject and set: id, label, color, centroid,
            #           convex_hull, valid, and reliability flags
            #         - Append to the DetectedObjectArray and publish after the loop
            # Calculate centroid from points3d 
            centroid = np.mean(points3d[:, :3], axis=0)

            # Calculate convex hull using 2D points. Z-axis for the hull is set to the minimum z of the cluster (bottom of the object)
            points_2d = MultiPoint(points3d[:, :2])
            hull = points_2d.convex_hull
            # The hull is a polygon only for 3+ non-collinear points; a line or a point is not a usable obstacle outline
            if hull.geom_type != "Polygon":
                continue
            min_z = float(np.min(points3d[:, 2]))

            # Hull coordinates as (N, 2) array; drop the last point — in a closed ring it duplicates the first one
            hull_points_2d = np.array(hull.exterior.coords)[:-1]
            # Append min_z as the z-coordinate to every hull point and flatten into [x1, y1, z1, x2, y2, z2, ...]
            convex_hull_points = np.hstack((hull_points_2d, np.full((len(hull_points_2d), 1), min_z))).ravel().tolist()

            # Create a DetectedObject and populate its fields
            obj = DetectedObject()
            obj.id = i
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            
            # Assign calculated centroid coordinates
            obj.centroid.x = float(centroid[0])
            obj.centroid.y = float(centroid[1])
            obj.centroid.z = float(centroid[2])
            
            obj.convex_hull = convex_hull_points
            obj.position_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False

            # Append to the DetectedObjectArray
            result_object_array.objects.append(obj)

        # Publish after the loop is completely finished building the array
        self.objects_pub.publish(result_object_array)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()