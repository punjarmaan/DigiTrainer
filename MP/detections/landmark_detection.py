import cv2
import mediapipe as mp
import numpy as np
import csv


LANDMARKS = 33
    
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
    
    if right_viz - left_viz > 0.1:
        return 2
    elif left_viz - right_viz > 0.1:
        return 1
    else:
        return 0 

def main():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    vid_path = ['bicep_bad1.MOV', 'bicep_bad2.MOV', 'bicep_bad3.MOV', 'bicep_bad4.MOV', 
                'bicep_bad5.MOV', 'bicep_bad6.MOV', 'bicep_bad7.MOV', 'bicep_bad8.mp4',
                'bicep_bad9.mp4', 'bicep_bad10.mp4', 'bicep_bad11.mp4',
                'bicep_good1.MOV', 'bicep_good2.MOV', 'bicep_good3.MOV', 'bicep_good4.mp4',
                'bicep_good5.mp4', 'bicep_good6.mp4', 'bicep_good7.mp4']

    landmark_labels = ['class']
    for val in range(1, LANDMARKS+1):
        landmark_labels += [f'x{val}', f'y{val}', f'z{val}']
    
    with open('../bicep_outputs/all_landmarks.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmark_labels)

    for i in range(len(vid_path)):
        cap = cv2.VideoCapture(vid_path[i])
        
        if vid_path[i] == 'bicep_bad1.MOV':
            label = 'bad'
        elif vid_path[i] == 'bicep_good1.MOV':
            label = 'good'

        ## Mediapipe pose instance
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
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

                    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

                    right_side = [r_shoulder, r_elbow, r_wrist]
                    left_side = [l_shoulder, l_elbow, l_wrist]

                    side = detect_side(right_side, left_side)

                    if side == 2:
                        curl_angle = calculate_angle([r_shoulder.x, r_shoulder.y, r_shoulder.z], 
                                                        [r_elbow.x, r_elbow.y, r_elbow.z], 
                                                        [r_wrist.x, r_wrist.y, r_wrist.z])
                        
                        upper_arm_torso_angle = calculate_angle([r_elbow.x, r_elbow.y, r_elbow.z],
                                                                [r_shoulder.x, r_shoulder.y, r_shoulder.z],
                                                                [r_hip.x, r_hip.y, r_hip.z])
                        torso_lean_angle = calculate_angle([r_shoulder.x, r_shoulder.y, r_shoulder.z],
                                                            [r_hip.x, r_hip.y, r_hip.z],
                                                            [r_knee.x, r_knee.y, r_knee.z])
                        
                        # print(f'Right side detected; CA: {curl_angle}; UAT: {upper_arm_torso_angle}; TL: {torso_lean_angle}')
                        
                        # cv2.putText(image, 'Right Detected', (20,20), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                        # cv2.putText(image, f'Right angle: {str(curl_angle)}', (20,50), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                    elif side == 1:
                        curl_angle = calculate_angle([l_shoulder.x, l_shoulder.y, l_shoulder.z], 
                                                        [l_elbow.x, l_elbow.y, l_elbow.z], 
                                                        [l_wrist.x, l_wrist.y, l_wrist.z])
                        
                        upper_arm_torso_angle = calculate_angle([l_elbow.x, l_elbow.y, l_elbow.z],
                                                                [l_shoulder.x, l_shoulder.y, l_shoulder.z],
                                                                [l_hip.x, l_hip.y, l_hip.z])
                        torso_lean_angle = calculate_angle([l_shoulder.x, l_shoulder.y, l_shoulder.z],
                                                            [l_hip.x, l_hip.y, l_hip.z],
                                                            [l_knee.x, l_knee.y, l_knee.z])
                        
                        # print(f'Left side detected; CA: {curl_angle}; UAT: {upper_arm_torso_angle}; TL: {torso_lean_angle}')
                        # cv2.putText(image, 'Left Detected', (20,20), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                        # cv2.putText(image, f'Left angle: {str(curl_angle)}', (20,50), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    else:
                        
                        l_curl_angle = calculate_angle([l_shoulder.x, l_shoulder.y, l_shoulder.z], 
                                                        [l_elbow.x, l_elbow.y, l_elbow.z], 
                                                        [l_wrist.x, l_wrist.y, l_wrist.z])
                        l_upper_arm_torso_angle = calculate_angle([l_elbow.x, l_elbow.y, l_elbow.z],
                                                                [l_shoulder.x, l_shoulder.y, l_shoulder.z],
                                                                [l_hip.x, l_hip.y, l_hip.z])
                        l_torso_lean_angle = calculate_angle([l_shoulder.x, l_shoulder.y, l_shoulder.z],
                                                            [l_hip.x, l_hip.y, l_hip.z],
                                                            [l_knee.x, l_knee.y, l_knee.z])
                        
                        r_curl_angle = calculate_angle([r_shoulder.x, r_shoulder.y, r_shoulder.z], 
                                                        [r_elbow.x, r_elbow.y, r_elbow.z], 
                                                        [r_wrist.x, r_wrist.y, r_wrist.z])
                        r_upper_arm_torso_angle = calculate_angle([r_elbow.x, r_elbow.y, r_elbow.z],
                                                                [r_shoulder.x, r_shoulder.y, r_shoulder.z],
                                                                [r_hip.x, r_hip.y, r_hip.z])
                        r_torso_lean_angle = calculate_angle([r_shoulder.x, r_shoulder.y, r_shoulder.z],
                                                            [r_hip.x, r_hip.y, r_hip.z],
                                                            [r_knee.x, r_knee.y, r_knee.z])
                        
                        # print(f'Middle detected;\n\tLCA: {l_curl_angle}; RCA{r_curl_angle}; \n\tLUAT: {l_upper_arm_torso_angle}; RUAT: {r_upper_arm_torso_angle}; \n\tLTL: {l_torso_lean_angle}; RTL: {r_torso_lean_angle}')


                        # cv2.putText(image, 'Both Arms Detected', (20,20), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                        # cv2.putText(image, f'Right angle: {str(r_curl_angle)}', (20,50), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                        # cv2.putText(image, f'Left angle: {str(l_curl_angle)}', (20,80), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                    
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
                    pose_row.insert(0, label)

                    with open('../bicep_outputs/all_landmarks.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(pose_row)


                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break 
            

            # formatted_data = list(np.array(data).flatten())
            # formatted_data.insert(0, label)

            # np.save(f'../bicep_outputs/bicep_bad{i+1}.npy', formatted_data)
            # print(formatted_data[0:4])

            cap.release()
            cv2.destroyAllWindows

main()