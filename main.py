from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

# model_1 = YOLO("yolov8n-pose.pt")

# video_path = "D:/computer_vision_project/_test/Zhuanti/_back_fall_1_short.mp4"

# results = model_1(source=video_path, show=True, conf=0.3, save=True)


model_1 = YOLO("yolov8n-pose.pt")
model_2 = YOLO(
    "D:\computer_vision_project\_runs\detect\_train3\weights\_best.pt")
# Path to the input video
video_path = "D:/computer_vision_project/_test/Zhuanti/fake_fall_real_af.mp4"
# Process video
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
pixel_in_m = 0.00018
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('D:/computer_vision_project/_test/_back_fall_1_results.mp4',
                      fourcc, 30.0, (frame_width, frame_height))


def main():
    x = {}
    y = {}
    v = {}
    t = {}
    k = 0
    j = 0
    figure, axis = plt.subplots(5, 2)
    figure_2, axis_2 = plt.subplots(5, 2)
    keypoints = []
    current_frame = 0
    falling_count = 0
    while True:
        current_frame += 1
        ret, frame = cap.read()
        if not ret:
            break
        # Perform pose estimation on the frame
        results_1 = model_1.track(frame)
        results_2 = model_2.track(frame)
        # Extract keypoints
        keypoints = results_1[0].keypoints.xy.cpu().tolist()
        # print(len(keypoints[0]))
        current_time = current_frame/frame_rate
        cv2.putText(frame, f'Time: {current_time:.0f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print(f'keypoint is {keypoints[0]}')
        for i in keypoints[0]:
            # print(k)
            if j == 0:
                if k == 0:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_0 = 0
                v_0 = []
                t_0 = []
                x_0 = []
                y_0 = []
                (max_v_0, v_0, t_0, x_0, y_0) = pose(
                    i, frame, current_frame, max_v_0, x[j], y[j], v[j], t[j], j)
                axis[0, 0].plot(t_0, v_0, color=(0.0000, 0.4470, 0.7410))
                axis[0, 0].set_title('head [0]')
                axis[0, 0].set_xlabel('second')
                axis[0, 0].set_ylabel('velocity')
                axis_2[0, 0].plot(t_0, y_0, color=(0.0000, 0.4470, 0.7410))
                axis_2[0, 0].set_title('head [0]')
                axis_2[0, 0].set_xlabel('second')
                axis_2[0, 0].set_ylabel('disp')
            if j == 7:
                if k == 7:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_7 = 0
                v_7 = []
                t_7 = []
                x_7 = []
                y_7 = []
                (max_v_7, v_7, t_7, x_7, y_7) = pose(
                    i, frame, current_frame, max_v_7, x[j], y[j], v[j], t[j], j)
                axis[1, 0].plot(t_7, v_7, color=(0.8500, 0.3250, 0.0980))
                axis[1, 0].set_title('left elbow [7]')
                axis[1, 0].set_xlabel('second')
                axis[1, 0].set_ylabel('velocity')
                axis_2[1, 0].plot(t_7, y_7, color=(0.8500, 0.3250, 0.0980))
                axis_2[1, 0].set_title('left elbow [7]')
                axis_2[1, 0].set_xlabel('second')
                axis_2[1, 0].set_ylabel('disp')
            if j == 8:
                if k == 8:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_8 = 0
                v_8 = []
                t_8 = []
                x_8 = []
                y_8 = []
                (max_v_8, v_8, t_8, x_8, y_8) = pose(
                    i, frame, current_frame, max_v_8, x[j], y[j], v[j], t[j], j)
                axis[2, 0].plot(t_8, v_8, color=(0.9290, 0.6940, 0.1250))
                axis[2, 0].set_title('right elbow [8]')
                axis[2, 0].set_xlabel('second')
                axis[2, 0].set_ylabel('velocity')
                axis_2[2, 0].plot(t_8, y_8, color=(0.9290, 0.6940, 0.1250))
                axis_2[2, 0].set_title('right elbow [8]')
                axis_2[2, 0].set_xlabel('second')
                axis_2[2, 0].set_ylabel('disp')
                print(current_frame)
                print(y_8)
                print(current_time)
                # if current_time > 5:
                #    time.sleep(1)
            if j == 9:
                if k == 9:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_9 = 0
                v_9 = []
                t_9 = []
                x_9 = []
                y_9 = []
                (max_v_9, v_9, t_9, x_9, y_9) = pose(
                    i, frame, current_frame, max_v_9, x[j], y[j], v[j], t[j], j)
                axis[3, 0].plot(t_9, v_9, color=(0.4940, 0.1840, 0.5560))
                axis[3, 0].set_title('left palm [9]')
                axis[3, 0].set_xlabel('second')
                axis[3, 0].set_ylabel('velocity')
                axis_2[3, 0].plot(t_9, y_9, color=(0.4940, 0.1840, 0.5560))
                axis_2[3, 0].set_title('left palm [9]')
                axis_2[3, 0].set_xlabel('second')
                axis_2[3, 0].set_ylabel('disp')
            if j == 10:
                if k == 10:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_10 = 0
                v_10 = []
                t_10 = []
                x_10 = []
                y_10 = []
                (max_v_10, v_10, t_10, x_10, y_10) = pose(
                    i, frame, current_frame, max_v_10, x[j], y[j], v[j], t[j], j)
                axis[4, 0].plot(t_10, v_10, color=(0.4660, 0.6740, 0.1880))
                axis[4, 0].set_title('right palm [10]')
                axis[4, 0].set_xlabel('second')
                axis[4, 0].set_ylabel('velocity')
                axis_2[4, 0].plot(t_10, y_10, color=(0.4660, 0.6740, 0.1880))
                axis_2[4, 0].set_title('right palm [10]')
                axis_2[4, 0].set_xlabel('second')
                axis_2[4, 0].set_ylabel('disp')
            if j == 13:
                if k == 13:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_13 = 0
                v_13 = []
                t_13 = []
                x_13 = []
                y_13 = []
                (max_v_13, v_13, t_13, x_13, y_13) = pose(
                    i, frame, current_frame, max_v_13, x[j], y[j], v[j], t[j], j)
                axis[0, 1].plot(t_13, v_13, color=(0.6350, 0.0780, 0.1840))
                axis[0, 1].set_title('right knee [13]')
                axis[0, 1].set_xlabel('second')
                axis[0, 1].set_ylabel('velocity')
                axis_2[0, 1].plot(t_13, y_13, color=(0.6350, 0.0780, 0.1840))
                axis_2[0, 1].set_title('right knee [13]')
                axis_2[0, 1].set_xlabel('second')
                axis_2[0, 1].set_ylabel('disp')
            if j == 14:
                if k == 14:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_14 = 0
                v_14 = []
                t_14 = []
                x_14 = []
                y_14 = []
                (max_v_14, v_14, t_14, x_14, y_14) = pose(
                    i, frame, current_frame, max_v_14, x[j], y[j], v[j], t[j], j)
                axis[1, 1].plot(t_14, v_14, color=(0.5020, 0.5020, 0.0000))
                axis[1, 1].set_title('left knee [14]')
                axis[1, 1].set_xlabel('second')
                axis[1, 1].set_ylabel('velocity')
                axis_2[1, 1].plot(t_14, y_14, color=(0.5020, 0.5020, 0.0000))
                axis_2[1, 1].set_title('left knee [14]')
                axis_2[1, 1].set_xlabel('second')
                axis_2[1, 1].set_ylabel('disp')
            if j == 15:
                if k == 15:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_15 = 0
                v_15 = []
                t_15 = []
                x_15 = []
                y_15 = []
                (max_v_15, v_15, t_15, x_15, y_15) = pose(
                    i, frame, current_frame, max_v_15, x[j], y[j], v[j], t[j], j)
                axis[2, 1].plot(t_15, v_15, color=(0.5647, 0.9333, 0.5647))
                axis[2, 1].set_title('left ankle [15]')
                axis[2, 1].set_xlabel('second')
                axis[2, 1].set_ylabel('velocity')
                axis_2[2, 1].plot(t_15, y_15, color=(0.5647, 0.9333, 0.5647))
                axis_2[2, 1].set_title('left ankle [15]')
                axis_2[2, 1].set_xlabel('second')
                axis_2[2, 1].set_ylabel('disp')
            if j == 16:
                if k == 16:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_16 = 0
                v_16 = []
                t_16 = []
                x_16 = []
                y_16 = []
                (max_v_16, v_16, t_16, x_16, y_16) = pose(
                    i, frame, current_frame, max_v_16, x[j], y[j], v[j], t[j], j)
                axis[3, 1].plot(t_16, v_16, color=(0.3010, 0.7450, 0.9330))
                axis[3, 1].set_title('right ankle [16]')
                axis[3, 1].set_xlabel('second')
                axis[3, 1].set_ylabel('velocity')
                axis_2[3, 1].plot(t_16, y_16, color=(0.3010, 0.7450, 0.9330))
                axis_2[3, 1].set_title('right ankle [16]')
                axis_2[3, 1].set_xlabel('second')
                axis_2[3, 1].set_ylabel('disp')
            if j == 0 or j == 7 or j == 8 or j == 9 or j == 10 or j == 13 or j == 14 or j == 15 or j == 16:
                if k == 0 or k == 7 or k == 8 or k == 9 or k == 10 or k == 13 or k == 14 or k == 15 or k == 16:
                    x[j] = []
                    y[j] = []
                    v[j] = []
                    t[j] = []
                    max_v_all = 0
                v_all = []
                t_all = []
                x_all = []
                y_all = []
                (max_v_all, v_all, t_all, x_all, y_all) = pose(
                    i, frame, current_frame, max_v_all, x[j], y[j], v[j], t[j], j)
                if j == 0:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.0000, 0.4470, 0.7410))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(t_all, y_all)
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 7:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.8500, 0.3250, 0.0980))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.8500, 0.3250, 0.0980))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 8:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.9290, 0.6940, 0.1250))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.9290, 0.6940, 0.1250))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 9:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.4940, 0.1840, 0.5560))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.4940, 0.1840, 0.5560))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 10:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.4660, 0.6740, 0.1880))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.4660, 0.6740, 0.1880))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 13:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.6350, 0.0780, 0.1840))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.6350, 0.0780, 0.1840))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 14:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.5020, 0.5020, 0.0000))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.5020, 0.5020, 0.0000))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 15:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.5647, 0.9333, 0.5647))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.5647, 0.9333, 0.5647))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
                if j == 16:
                    axis[4, 1].plot(t_all, v_all, color=(
                        0.3010, 0.7450, 0.9330))
                    axis[4, 1].set_title('all joints')
                    axis[4, 1].set_xlabel('second')
                    axis[4, 1].set_ylabel('velocity')
                    axis_2[4, 1].plot(
                        t_all, y_all, color=(0.3010, 0.7450, 0.9330))
                    axis_2[4, 1].set_title('all joints')
                    axis_2[4, 1].set_xlabel('second')
                    axis_2[4, 1].set_ylabel('disp')
            j += 1
            k += 1
        for box, conf in zip(results_2[0].boxes.xyxy.tolist(), results_2[0].boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            max_v = max(max_v_0, max_v_7, max_v_8, max_v_9,
                        max_v_10, max_v_13, max_v_14, max_v_15, max_v_16)
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'fall detected, max velocity is {max_v:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                falling_count = falling_count + 1
        j = 0
        # print(f'j after zeroing {j}')
        cv2.imshow('Pose Estimation', frame)
    # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        # time.sleep(0.5)
    # print(f'left ankle position {y[15]}')
    # print(f'left ankle array {v[15]}')
    # print(f'right ankle position {y[16]}')
    # print(f'right ankle array {v[16]}')
    print(f'max v of keypoint 0 is {max_v_0}')
    print(f'max v of keypoint 7 is {max_v_7}')
    print(f'max v of keypoint 8 is {max_v_8}')
    print(f'max v of keypoint 9 is {max_v_9}')
    print(f'max v of keypoint 10 is {max_v_10}')
    print(f'max v of keypoint 13 is {max_v_13}')
    print(f'max v of keypoint 14 is {max_v_14}')
    print(f'max v of keypoint 15 is {max_v_15}')
    print(f'max v of keypoint 16 is {max_v_16}')
    if falling_count != 0:
        max_v = max(max_v_0, max_v_7, max_v_8, max_v_9,
                    max_v_10, max_v_13, max_v_14, max_v_15, max_v_16)
        print(f'the max fall velocity is {max_v}')
        if max_v >= 13.4:
            print("call 119 immediately, dont wait!!!")
        if max_v > 10 and max_v < 13.4:
            # in m/s
            print("lethal injury, call 119")
        else:
            print("no lethal injury but please becareful")
    else:
        print('no fall detected')
    # plt.title('velocity vs Frame')
    # plt.xlabel('Frame')
    # plt.ylabel('velocity')
    plt.show()
    # Show the plot
    # Release the video capture object and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def pose(i, frame, current_frame, max_v, x, y, v, t, j):
    x_val, y_val = i
    # cv2.circle(frame, (int(x_val), int(y_val)), 5, (0, 0, 255), -1)
    # cv2.putText(frame, f'{j}', (int(x_val), int(y_val)),
    #            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # Update plot
    x.append(x_val)
    y.append(y_val)
    # print(x)
    # print(y)
    # plt.plot(x, y, color='b'
    t.append(current_frame/frame_rate)
    previous_y = y[len(y)-2]
    current_y = y[len(y)-1]
    previous_t = t[len(t)-2]
    current_t = t[len(t)-1]
    if current_y-previous_y != 0:
        velocity = (current_y-previous_y)/(current_t-previous_t)
    else:
        velocity = 0
    # print(frame_rate)
    v.append(abs(velocity*frame_rate*pixel_in_m))
    current_v = v[len(v)-1]
    prev_v = v[len(v)-2]
    # print(v)
    if previous_y != 0 and prev_v != 0 and current_y != 0:
        if max_v < current_v:
            max_v = current_v
    if current_v > 20:
        v[len(v)-1] = 0

    # print(f"coor x is {x}, coor y is {y}")
    # k += 1
    # cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    # k = 0
    return (max_v, v, t, x, y)


main()
