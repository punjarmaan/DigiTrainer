import cv2
import mediapipe as mp
import numpy as np
import csv
import time

LANDMARKS = 33 #MP outputs 33 3D landmarks
    
## Finds angle between 3 3d points
def calculate_angle(a, b, c):

    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_ang))

    return angle

## Detects which arm is most prominent
def detect_side(right, left):
    right_viz = 0
    left_viz = 0
    i = 0

    while i < 3:
        right_viz += (right[i].visibility / 3)
        left_viz += (left[i].visibility / 3)
        
        i += 1
    
    if right_viz > left_viz:
        return 1
    else:
        return 0
    
## Detects most prominent side and get the angles from that side
def getAngles(right, left):
    side = detect_side(right, left)

    angles = []

    if side:
        # Right side is more prominent
        shoulder = right[0]
        elbow = right[1]
        wrist = right[2]
        hip = right[3]
        knee = right[4]
        index = right[5]

        curl_angle = calculate_angle([shoulder.x, shoulder.y, shoulder.z], 
                                    [elbow.x, elbow.y, elbow.z], 
                                    [wrist.x, wrist.y, wrist.z])
        upper_arm_torso_angle = calculate_angle([elbow.x, elbow.y, elbow.z],
                                            [shoulder.x, shoulder.y, shoulder.z],
                                            [hip.x, hip.y, hip.z])
        torso_lean_angle = calculate_angle([shoulder.x, shoulder.y, shoulder.z],
                                            [hip.x, hip.y, hip.z],
                                            [knee.x, knee.y, knee.z])
        wrist_flexion_angle = calculate_angle([elbow.x, elbow.y, elbow.z],
                                              [wrist.x, wrist.y, wrist.z],
                                              [index.x, index.y, index.z])
        
        angles.extend([curl_angle, upper_arm_torso_angle, torso_lean_angle, wrist_flexion_angle])
    else:
        #Left side is more visible
        shoulder = left[0]
        elbow = left[1]
        wrist = left[2]
        hip = left[3]
        knee = left[4]
        index = left[5]

        curl_angle = calculate_angle([shoulder.x, shoulder.y, shoulder.z], 
                                    [elbow.x, elbow.y, elbow.z], 
                                    [wrist.x, wrist.y, wrist.z])
        upper_arm_torso_angle = calculate_angle([elbow.x, elbow.y, elbow.z],
                                            [shoulder.x, shoulder.y, shoulder.z],
                                            [hip.x, hip.y, hip.z])
        torso_lean_angle = calculate_angle([shoulder.x, shoulder.y, shoulder.z],
                                            [hip.x, hip.y, hip.z],
                                            [knee.x, knee.y, knee.z])
        wrist_flexion_angle = calculate_angle([elbow.x, elbow.y, elbow.z],
                                              [wrist.x, wrist.y, wrist.z],
                                              [index.x, index.y, index.z])
        
        angles.extend([curl_angle, upper_arm_torso_angle, torso_lean_angle, wrist_flexion_angle])
    
    return angles

def save_vid_data(file, vals):
    with open(file, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(vals)

def main():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    vid_path = ['bicep_bad1.MOV', 'bicep_bad2.MOV', 'bicep_bad3.MOV', 'bicep_bad4.MOV', 
                'bicep_bad5.MOV', 'bicep_bad6.MOV', 'bicep_bad7.MOV', 'bicep_bad8.mp4',
                'bicep_bad9.mp4', 'bicep_bad10.mp4', 'bicep_bad11.mp4',
                'bicep_good1.MOV', 'bicep_good2.MOV', 'bicep_good3.MOV', 'bicep_good4.mp4',
                'bicep_good5.mp4', 'bicep_good6.mp4', 'bicep_good7.mp4']
    
    vid_path = ['bicep_good6.mp4']


    for i in range(len(vid_path)):
        cap = cv2.VideoCapture(vid_path[i])
        
        if vid_path[i] == 'bicep_bad1.MOV':
            label = 0
        elif vid_path[i] == 'bicep_good1.MOV':
            label = 1

        label = 1
        
        curl = [vid_path[i], label]
        upper_arm_torso = [vid_path[i], label]
        torso_lean = [vid_path[i], label]
        wrist_flexion = [vid_path[i], label]

        ## Mediapipe pose instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                time.sleep(1)
                ret, frame = cap.read()
                if not ret:
                    print("Empty camera frame detected.")
                    break

                # Recolor image to be RGB - needed for MP
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                

                # Makes detection
                result = pose.process(image)

                # # Recolor back to BGR - needed for CV2
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                # Drawing Specs
                mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                
                # Show Image 
                cv2.imshow('Mediapipe Pose Estimation', image)
                #  cv2.rectangle(image, (0,0), (150,100), (255,255,255), 1)
                
                
                # Derive Landmarks
                if not result.pose_landmarks:
                    # cv2.putText(image, 'No Arms Detected', (20,20), 
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                    pass
                else:
                    landmarks = result.pose_landmarks.landmark

                    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    r_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]

                    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    l_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                    

                    right_side = [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_index]
                    left_side = [l_shoulder, l_elbow, l_wrist, l_hip, l_knee, l_index]

                    angles = getAngles(right_side, left_side)

                    curl.append(angles[0])
                    upper_arm_torso.append(angles[1])
                    torso_lean.append(angles[2])
                    wrist_flexion.append(angles[3])


                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break 
            

            # formatted_data = list(np.array(data).flatten())
            # formatted_data.insert(0, label)

            # np.save(f'../bicep_outputs/bicep_bad{i+1}.npy', formatted_data)
            # print(formatted_data[0:4])

            cap.release()

        # save_vid_data('../bicep_outputs/curl_angles.csv', curl)
        # save_vid_data('../bicep_outputs/upper_arm_torso_angles.csv', upper_arm_torso)
        # save_vid_data('../bicep_outputs/torso_lean_angles.csv', torso_lean)
        # save_vid_data('../bicep_outputs/wrist_flexion_angles.csv', wrist_flexion)

main()