"""Record live Gazebo timing, validity and mesh-ray surface error without using depth as truth."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
from cv_bridge import CvBridge
import numpy as np
from opl_interfaces.msg import DetectedObjectArray
from diagnostic_msgs.msg import DiagnosticArray
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose_matrix(text):
    x, y, z, roll, pitch, yaw = map(float, (text or '0 0 0 0 0 0').split())
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    result[:3, 3] = [x, y, z]
    return result


def triangles(world):
    root = ET.parse(world).find('world')
    result = {}
    for model in root.findall('model'):
        name = model.get('name')
        cls = next((c for c in ['bottle', 'cup', 'book', 'chair'] if c in name), None)
        if cls is None:
            continue
        pieces = []
        for link in model.findall('link'):
            for visual in link.findall('visual'):
                transform = pose_matrix(model.findtext('pose')) @ pose_matrix(link.findtext('pose')) @ pose_matrix(visual.findtext('pose'))
                mesh, box = visual.find('geometry/mesh'), visual.find('geometry/box')
                vertices, faces = [], []
                if mesh is not None:
                    uri = mesh.findtext('uri')
                    path = Path(world).parent / 'models' / uri.removeprefix('model://')
                    if path.suffix.lower() != '.obj':
                        continue
                    for line in path.read_text().splitlines():
                        parts = line.split()
                        if parts and parts[0] == 'v':
                            vertices.append(list(map(float, parts[1:4])))
                        elif parts and parts[0] == 'f':
                            indices = [int(p.split('/')[0]) for p in parts[1:]]
                            indices = [i-1 if i > 0 else len(vertices)+i for i in indices]
                            faces.extend([[indices[0], indices[j], indices[j+1]] for j in range(1, len(indices)-1)])
                    vertices = np.asarray(vertices) * np.array(list(map(float, (mesh.findtext('scale') or '1 1 1').split())))
                elif box is not None:
                    size = np.array(list(map(float, box.findtext('size').split()))) / 2
                    vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * size
                    faces = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                             [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]]
                if len(vertices) and faces:
                    vertices = np.asarray(vertices) @ transform[:3,:3].T + transform[:3,3]
                    pieces.append(vertices[np.asarray(faces)])
        if pieces:
            result[name] = (cls, np.concatenate(pieces))
    return result


def ray_surface(origin, direction, mesh):
    edge1, edge2 = mesh[:,1]-mesh[:,0], mesh[:,2]-mesh[:,0]
    h = np.cross(direction, edge2)
    determinant = np.einsum('ij,ij->i', edge1, h)
    valid = abs(determinant) > 1e-10
    inv = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    s = origin-mesh[:,0]
    u = inv*np.einsum('ij,ij->i', s, h)
    q = np.cross(s, edge1)
    v = inv*(q@direction)
    distance = inv*np.einsum('ij,ij->i', edge2, q)
    valid &= (u>=0)&(v>=0)&(u+v<=1)&(distance>0)
    return float(distance[valid].min()) if valid.any() else None


def self_check():
    mesh = np.array([[[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]]])
    assert ray_surface(np.zeros(3), np.array([0., 0., 1.]), mesh) == 2.
    assert ray_surface(np.zeros(3), np.array([0., 0., -1.]), mesh) is None
    assert np.allclose(rotation([0., 0., 0., 1.]), np.eye(3))
    assert np.allclose(pose_matrix('1 2 3 0 0 0')[:3, 3], [1, 2, 3])


def main():
    self_check()
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--output', required=True)
    parser.add_argument('--poses', required=True)
    parser.add_argument('--world', required=True)
    args = parser.parse_args()
    meshes = triangles(args.world)
    poses = json.loads(Path(args.poses).read_text())
    robot = next(p for p in poses['pose'] if p['name']=='wheeltec_oplbot')
    world_r = rotation([robot['orientation'].get(k, 0.) for k in ['x','y','z','w']])
    world_t = np.array([robot['position'].get(k,0.) for k in ['x','y','z']])
    rclpy.init()
    node = rclpy.create_node('object_live_measurement', parameter_overrides=[Parameter('use_sim_time', value=True)])
    buffer = Buffer(node=node)
    listener = TransformListener(buffer,node)
    counts, classes, colors, statuses = Counter(), Counter(), Counter(), Counter()
    images, depths, infos = {}, {}, {}
    metrics, samples, surface_errors, matched_surfaces = [], [], {}, Counter()
    tf_errors, alignment, camera_sample = [], [], []
    clock_samples = []
    last_image = [None]
    subscriptions = []
    def stamp(header):
        return header.stamp.sec*10**9+header.stamp.nanosec
    def camera(topic, msg, cache):
        counts[topic] += 1
        cache[stamp(msg.header)] = msg
        if len(cache)>60:
            del cache[next(iter(cache))]
        if topic=='rgb':
            last_image[0] = msg
    def detections(msg):
        counts['detections'] += 1
        key = stamp(msg.header)
        if key in images and key in depths and key in infos:
            alignment.append(images[key].header.frame_id==depths[key].header.frame_id==infos[key].header.frame_id)
            if not camera_sample:
                camera_sample.append({'rgb_stamp_ns':key,'depth_stamp_ns':stamp(depths[key].header),
                    'info_stamp_ns':stamp(infos[key].header),'rgb_frame':images[key].header.frame_id,
                    'depth_frame':depths[key].header.frame_id,'info_frame':infos[key].header.frame_id,
                    'width':images[key].width,'height':images[key].height,'rgb_encoding':images[key].encoding,
                    'depth_encoding':depths[key].encoding,'K':list(infos[key].k),'D':list(infos[key].d)})
        for obj in msg.objects:
            classes[obj.class_name] += 1
            colors[obj.class_name+':'+obj.color] += 1
            statuses[obj.class_name+':'+obj.position_status] += 1
            assert not obj.orientation_valid and not obj.fixed_orientation_valid
            if len(samples)<20:
                p=obj.camera_pose.pose.position
                samples.append({'id':obj.detection_id,'class':obj.class_name,'confidence':obj.confidence,
                                'color':obj.color,'color_confidence':obj.color_confidence,
                                'camera_xyz':[p.x,p.y,p.z], 'position_valid':obj.position_valid,
                                'position_status':obj.position_status,'fixed_valid':obj.fixed_position_valid,
                                'fixed_status':obj.fixed_status,'stamp_ns':key,'frame':msg.header.frame_id})
            if not obj.position_valid:
                continue
            p=obj.camera_pose.pose.position
            point=np.array([p.x,p.y,p.z])
            try:
                transform=buffer.lookup_transform('base_footprint',msg.header.frame_id,rclpy.time.Time.from_msg(msg.header.stamp))
                q=transform.transform.rotation; t=transform.transform.translation
                local_r=rotation([q.x,q.y,q.z,q.w]); local_t=np.array([t.x,t.y,t.z])
                origin=world_t+world_r@local_t
                direction=world_r@local_r@(point/np.linalg.norm(point))
                hits=[]
                for name,(cls,mesh) in meshes.items():
                    if cls!=obj.class_name:
                        continue
                    distance=ray_surface(origin,direction,mesh)
                    if distance is not None:
                        hits.append((distance,name))
                if hits:
                    distance,name=min(hits)
                    surface_errors.setdefault(name,[]).append(abs(float(np.linalg.norm(point))-distance))
                    matched_surfaces[name] += 1
                if obj.fixed_position_valid:
                    fixed=buffer.lookup_transform(obj.fixed_pose.header.frame_id,msg.header.frame_id,rclpy.time.Time.from_msg(msg.header.stamp))
                    q=fixed.transform.rotation;t=fixed.transform.translation
                    expected=rotation([q.x,q.y,q.z,q.w])@point+np.array([t.x,t.y,t.z])
                    p2=obj.fixed_pose.pose.position
                    tf_errors.append(float(np.linalg.norm(expected-[p2.x,p2.y,p2.z])))
            except Exception:
                counts['measurement_tf_miss'] += 1
    def diagnostic(msg):
        if msg.header.frame_id=='head_cam_optical_link' and msg.status and msg.status[0].message=='ok':
            metrics.append({v.key:float(v.value) for v in msg.status[0].values})
    for topic,typ,callback in [('/head_cam/image',Image,lambda m:camera('rgb',m,images)),
                              ('/head_cam/depth_image',Image,lambda m:camera('depth',m,depths)),
                              ('/head_cam/camera_info',CameraInfo,lambda m:camera('info',m,infos)),
                              ('/detected_objects',DetectedObjectArray,detections),
                              ('/object_detection/diagnostics',DiagnosticArray,diagnostic),
                              ('/clock',Clock,lambda m:clock_samples.append((time.monotonic(),m.clock.sec+m.clock.nanosec*1e-9)))]:
        subscriptions.append(node.create_subscription(typ,topic,callback,qos_profile_sensor_data))
    start=time.monotonic()
    while time.monotonic()-start<args.duration:
        rclpy.spin_once(node,timeout_sec=.02)
    elapsed=time.monotonic()-start
    def summary(values):
        return {'count':len(values),'median':float(np.median(values)),'p95':float(np.percentile(values,95)),
                'max':float(max(values))} if values else {'count':0}
    report={'duration_wall_s':elapsed,'counts':dict(counts),'processed_wall_fps':counts['detections']/elapsed,
            'class_counts':dict(classes),'color_counts':dict(colors),'position_status_counts':dict(statuses),
            'camera_sample':camera_sample, 'clock_sample_count':len(clock_samples),
            'clock_monotonic':all(b[1]>=a[1] for a,b in zip(clock_samples,clock_samples[1:])),
            'exact_stamp_frame_matches':len(alignment),'all_matched_frames_equal':all(alignment),
            'sample_detections':samples,'surface_error_m':{name:summary(v) for name,v in surface_errors.items()},
            'tf_consistency_error_m':summary(tf_errors),
            'metrics':{key:summary([m[key] for m in metrics]) for key in ['inference_wall_ms','capture_to_publication_ros_ms',
                      'reception_to_publication_wall_ms','processing_wall_ms']},'last_metrics':metrics[-1] if metrics else {},
            'robot_world_pose':robot}
    if len(metrics)>1:
        report['node_counter_delta'] = {k:metrics[-1][k]-metrics[0][k] for k in
            ['received_frames','dropped_frames','processed_frames','inference_errors']}
    if len(clock_samples)>1:
        report['clock_real_time_factor']=(clock_samples[-1][1]-clock_samples[0][1])/(clock_samples[-1][0]-clock_samples[0][0])
    if last_image[0] is not None:
        cv2.imwrite(str(Path(args.output).with_suffix('.jpg')),CvBridge().imgmsg_to_cv2(last_image[0],'bgr8'))
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample_detections','robot_world_pose')},indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
