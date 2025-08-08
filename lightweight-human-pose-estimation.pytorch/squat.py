import cv2
import numpy as np
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from val import normalize, pad_width
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses
import time
import math

def resize(img, scale):
    h, w, _ = img.shape
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_img

def get_angle(a, b, c):
    """Calculate angle between 3 points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def is_squat_down(hip, knee, ankle):
    angle = get_angle(hip, knee, ankle)
    return angle < 100


def is_standing(hip, knee, ankle):
    angle = get_angle(hip, knee, ankle)
    return angle > 160


def main():
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    net = net.eval()

    cap = cv2.VideoCapture(0)

    height_size = 256
    stride = 8
    upsample_ratio = 4
    num_keypoints = 18
    pose_num = 0

    previous_poses = []
    squat_counter = 0
    squat_down = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_img = frame.copy()
        scale = height_size / orig_img.shape[0]
        scaled_img = resize(orig_img, scale)
        padded_img, pad = pad_width(scaled_img, stride, pad_value=(0, 0, 0))
        img_mean = np.array([128, 128, 128], np.float32)
        img_scale = np.float32(1/256)
        normalized_img = normalize(padded_img, img_mean, img_scale)
        tensor_img = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0)

        stages_output = net(tensor_img)
        stage2_heatmaps = stages_output[-2]
        stage2_pafs = stages_output[-1]

        heatmaps = stage2_heatmaps.squeeze().detach().numpy()
        pafs = stage2_pafs.squeeze().detach().numpy()

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            keypoints = extract_keypoints(heatmaps[kpt_idx], kpt_idx, total_keypoints_num)
            keypoints = list(keypoints)
            total_keypoints_num += len(keypoints)
            all_keypoints_by_type.append(keypoints)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs,
                                                      demo=True)

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                kp = pose_entries[n][kpt_id]
                if kp != -1.0:
                    pose_keypoints[kpt_id] = all_keypoints[int(kp), 0:2]
            current_poses.append(Pose(pose_keypoints, pose_entries[n][18]))

        current_poses = track_poses(previous_poses, current_poses, smooth=True)
        previous_poses = current_poses

        for pose in current_poses:
            # draw skeleton
            pose.draw(frame)

            # Squat keypoints: hip(8), knee(9), ankle(10)
            try:
                hip = pose.keypoints[8]
                knee = pose.keypoints[9]
                ankle = pose.keypoints[10]

                if -1 in hip or -1 in knee or -1 in ankle:
                    continue

                angle = get_angle(hip, knee, ankle)
                cv2.putText(frame, f"Angle: {int(angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                if is_squat_down(hip, knee, ankle):
                    if not squat_down:
                        squat_down = True
                elif is_standing(hip, knee, ankle):
                    if squat_down:
                        squat_counter += 1
                        squat_down = False

            except Exception as e:
                print("Error calculating angle:", e)

        cv2.putText(frame, f"Squat Count: {squat_counter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Squat Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
