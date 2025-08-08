import argparse
import cv2
import numpy as np
import torch
import socket
import json  # make sure this is imported for JSON conversion
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def run_demo(net, image_provider, height_size, cpu, track, smooth, exercise_type):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    pose_states = {}
    delay = 1

    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

            keypoints = pose.keypoints
            pose_id = pose.id
            
            # JSON dictionary for Unity
           # pose_json = {}
           # for i, (x, y) in enumerate(pose.keypoints):
            #    pose_json[str(i)] = [int(x), int(y)]  # convert floats to int

            #try:
             #   json_data = json.dumps(pose_json)
             #   conn.sendall(json_data.encode('utf-8') + b'\n')  # newline = end of message
            #except Exception as e:
            #    print("Socket send error:", e)



            if pose_id not in pose_states:
                pose_states[pose_id] = {
                    'reps': 0,
                    'bicep_phase': 'down',
                    'pushup_phase': 'up',
                    'squat_phase': 'up',
                     'last_phase_said': '' 
                }

            # Track bicep curls
            if exercise_type == 'bicep':
                shoulder, elbow, wrist = keypoints[2], keypoints[3], keypoints[4]
                if -1 in shoulder or -1 in elbow or -1 in wrist:
                    continue
                angle = calculate_angle(shoulder, elbow, wrist)
                if angle < 40 and pose_states[pose_id]['bicep_phase'] == 'down':
                    pose_states[pose_id]['bicep_phase'] = 'up'
                    if pose_states[pose_id]['last_phase_said'] != 'up':
                     speak("Up")
                     pose_states[pose_id]['last_phase_said'] = 'up'
                elif angle > 160 and pose_states[pose_id]['bicep_phase'] == 'up':
                    pose_states[pose_id]['bicep_phase'] = 'down'
                    speak("Down")
                    pose_states[pose_id]['reps'] += 1
                    speak(f"{pose_states[pose_id]['reps']} completed")

                status = "Bent" if angle < 40 else "Stretched" if angle > 160 else "Moving"
                cv2.putText(img, f'Elbow Angle: {int(angle)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f'Status: {status}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Track push-ups
            elif exercise_type == 'pushup':
                shoulder, elbow, wrist = keypoints[2], keypoints[3], keypoints[4]
                if -1 in shoulder or -1 in elbow or -1 in wrist:
                    continue
                angle = calculate_angle(shoulder, elbow, wrist)
                if angle < 80 and pose_states[pose_id]['pushup_phase'] == 'up':
                    pose_states[pose_id]['pushup_phase'] = 'down'
                elif angle > 160 and pose_states[pose_id]['pushup_phase'] == 'down':
                    pose_states[pose_id]['pushup_phase'] = 'up'
                    pose_states[pose_id]['reps'] += 1

                status = "Down" if angle < 80 else "Up" if angle > 160 else "Moving"
                cv2.putText(img, f'Elbow Angle: {int(angle)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f'Status: {status}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Track squats
            elif exercise_type == 'squat':
                hip, knee, ankle = keypoints[8], keypoints[9], keypoints[10]
                if -1 in hip or -1 in knee or -1 in ankle:
                    continue
                angle = calculate_angle(hip, knee, ankle)
                if angle < 120 and pose_states[pose_id]['squat_phase'] == 'up':
                    pose_states[pose_id]['squat_phase'] = 'down'
                elif angle > 160 and pose_states[pose_id]['squat_phase'] == 'down':
                    pose_states[pose_id]['squat_phase'] = 'up'
                    pose_states[pose_id]['reps'] += 1

                status = "Down" if angle < 90 else "Up" if angle > 160 else "Moving"
                cv2.putText(img, f'Knee Angle: {int(angle)} deg', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f'Status: {status}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show rep count
            cv2.putText(img, f'Reps: {pose_states[pose_id]["reps"]}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f'{exercise_type.capitalize()} Tracker', img)
        key = cv2.waitKey(delay)
        if key == 27:
            return
        elif key == 112:
            delay = 0 if delay == 1 else 1


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    stages_output = net(tensor_img)
    heatmaps = np.transpose(stages_output[-2].squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    pafs = np.transpose(stages_output[-1].squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    return heatmaps, pafs, scale, pad


class VideoReader:
    def __init__(self, file_name):
        try:
            self.file_name = int(file_name)
        except ValueError:
            self.file_name = file_name

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f'Video {self.file_name} cannot be opened')
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--height-size', type=int, default=256)
    parser.add_argument('--video', type=str, default='0')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--track', type=int, default=1)
    parser.add_argument('--smooth', type=int, default=1)
    args = parser.parse_args()

    print("Select exercise type:")
    print("1. Bicep Curl")
    print("2. Push-Up")
    print("3. Squat")
    choice = input("Enter choice (1/2/3): ").strip()

    exercise_map = {'1': 'bicep', '2': 'pushup', '3': 'squat'}
    exercise_type = exercise_map.get(choice)
    if exercise_type is None:
        print("Invalid selection.")
        exit()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = VideoReader(args.video)
    # Socket server setup
    #HOST = '127.0.0.1'  # localhost or your PC's IP
    #PORT = 9999         # any available port
    #server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #server_socket.bind((HOST, PORT))
    #server_socket.listen(1)
    #print(f"Waiting for Unity to connect on {HOST}:{PORT}...")
    #conn, addr = server_socket.accept()
    #print(f"Connected by {addr}")
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, exercise_type)
