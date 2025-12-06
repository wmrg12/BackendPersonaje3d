

import cv2
import mediapipe as mp
import numpy as np
import json
import math
from datetime import datetime

# INICIALIZACIÓN DE MEDIAPIPE
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# CONSTANTES
IDENTITY = [1.0, 0.0, 0.0, 0.0]
ZERO = [0.0, 0.0, 0.0]


# FUNCIONES DE CÁLCULO DE QUATERNIONS

def normalize_vector(v):
    """Normaliza un vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.array([0, 0, 1])
    return v / norm

def quaternion_desde_vectores(v_desde, v_hacia, factor_suavizado=1.0):
    """
    Calcula el quaternion 
    """
    v_desde = normalize_vector(np.array(v_desde))
    v_hacia = normalize_vector(np.array(v_hacia))
    
    dot = np.clip(np.dot(v_desde, v_hacia), -1.0, 1.0)
    

    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    

    if dot < -0.9999:
        axis = np.cross(np.array([1, 0, 0]), v_desde)
        if np.linalg.norm(axis) < 0.001:
            axis = np.cross(np.array([0, 1, 0]), v_desde)
        axis = normalize_vector(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    
    # suavizado
    axis = normalize_vector(np.cross(v_desde, v_hacia))
    angle = math.acos(dot) * factor_suavizado 
    
    w = math.cos(angle / 2)
    sin_half = math.sin(angle / 2)
    x, y, z = axis * sin_half
    
    return np.array([w, x, y, z])

def normalize_quaternion(q):
    """Normaliza un quaternion."""
    norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if norm < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    return [q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm]

def multiplicar_quaternions(q1, q2):
    """Multiplica dos quaternions (q1 * q2)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return normalize_quaternion([w, x, y, z])

# Rotacion
MIXAMO_BASE_ROTATION = [0.7071068, 0.0, -0.7071068, 0.0]  

# SUAVIZADO TEMPORAL 

previous_frame_data = None
TEMPORAL_SMOOTHING = 0.7  

def smooth_quaternion_temporal(current_quat, previous_quat, factor=TEMPORAL_SMOOTHING):
    """
    Suaviza un quaternion usando el frame anterior.
    """
    if previous_quat is None:
        return current_quat
    
    # Interpolación lineal entre frames 
    result = []
    for i in range(4):
        result.append(previous_quat[i] * factor + current_quat[i] * (1 - factor))
    
    return normalize_quaternion(result)


# CAPTURA

def capturar_frame(landmarks, previous_frame=None):
    """
    Captura un frame y calcula las rotaciones COMO MIXAMO.
    """
    
    # Inicializar frame con identidad
    frame_data = {}
    
    # huesos
    bones = [
        "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
        "mixamorig:Neck", "mixamorig:Head", "mixamorig:HeadTop_End",
        "mixamorig:LeftShoulder", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand",
        "mixamorig:LeftHandThumb1", "mixamorig:LeftHandThumb2", "mixamorig:LeftHandThumb3", "mixamorig:LeftHandThumb4",
        "mixamorig:LeftHandIndex1", "mixamorig:LeftHandIndex2", "mixamorig:LeftHandIndex3", "mixamorig:LeftHandIndex4",
        "mixamorig:LeftHandMiddle1", "mixamorig:LeftHandMiddle2", "mixamorig:LeftHandMiddle3", "mixamorig:LeftHandMiddle4",
        "mixamorig:LeftHandRing1", "mixamorig:LeftHandRing2", "mixamorig:LeftHandRing3", "mixamorig:LeftHandRing4",
        "mixamorig:LeftHandPinky1", "mixamorig:LeftHandPinky2", "mixamorig:LeftHandPinky3", "mixamorig:LeftHandPinky4",
        "mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm", "mixamorig:RightHand",
        "mixamorig:RightHandThumb1", "mixamorig:RightHandThumb2", "mixamorig:RightHandThumb3", "mixamorig:RightHandThumb4",
        "mixamorig:RightHandIndex1", "mixamorig:RightHandIndex2", "mixamorig:RightHandIndex3", "mixamorig:RightHandIndex4",
        "mixamorig:RightHandMiddle1", "mixamorig:RightHandMiddle2", "mixamorig:RightHandMiddle3", "mixamorig:RightHandMiddle4",
        "mixamorig:RightHandRing1", "mixamorig:RightHandRing2", "mixamorig:RightHandRing3", "mixamorig:RightHandRing4",
        "mixamorig:RightHandPinky1", "mixamorig:RightHandPinky2", "mixamorig:RightHandPinky3", "mixamorig:RightHandPinky4",
        "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftToeBase", "mixamorig:LeftToe_End",
        "mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot", "mixamorig:RightToeBase", "mixamorig:RightToe_End"
    ]
    
    for bone in bones:
        frame_data[bone] = {
            "rotation_quaternion": IDENTITY.copy(),
            "location": ZERO.copy()
        }
    
    # Extraer landmarks
    lm = landmarks.landmark
    
    l_shoulder = np.array([lm[11].x, -lm[11].y, lm[11].z])
    r_shoulder = np.array([lm[12].x, -lm[12].y, lm[12].z])
    l_elbow = np.array([lm[13].x, -lm[13].y, lm[13].z])
    r_elbow = np.array([lm[14].x, -lm[14].y, lm[14].z])
    l_wrist = np.array([lm[15].x, -lm[15].y, lm[15].z])
    r_wrist = np.array([lm[16].x, -lm[16].y, lm[16].z])
    l_hip = np.array([lm[23].x, -lm[23].y, lm[23].z])
    r_hip = np.array([lm[24].x, -lm[24].y, lm[24].z])
    l_knee = np.array([lm[25].x, -lm[25].y, lm[25].z])
    r_knee = np.array([lm[26].x, -lm[26].y, lm[26].z])
    l_ankle = np.array([lm[27].x, -lm[27].y, lm[27].z])
    r_ankle = np.array([lm[28].x, -lm[28].y, lm[28].z])
    
    # Manos
    l_index = np.array([lm[19].x, -lm[19].y, lm[19].z]) if len(lm) > 19 else l_wrist
    r_index = np.array([lm[20].x, -lm[20].y, lm[20].z]) if len(lm) > 20 else r_wrist
    l_pinky = np.array([lm[17].x, -lm[17].y, lm[17].z]) if len(lm) > 17 else l_wrist
    r_pinky = np.array([lm[18].x, -lm[18].y, lm[18].z]) if len(lm) > 18 else r_wrist
    
    # Caderas
    if l_hip is not None and r_hip is not None:
        # Vector de caderas
        hip_vector = r_hip - l_hip
        if np.linalg.norm(hip_vector) > 0.01:
            movement_rotation = quaternion_desde_vectores(np.array([1, 0, 0]), hip_vector, factor_suavizado=0.03)
            hips_rotation = multiplicar_quaternions(MIXAMO_BASE_ROTATION, movement_rotation)
            frame_data["mixamorig:Hips"]["rotation_quaternion"] = normalize_quaternion(hips_rotation)
        else:

            frame_data["mixamorig:Hips"]["rotation_quaternion"] = MIXAMO_BASE_ROTATION

        frame_data["mixamorig:Hips"]["location"] = [0.0, 0.0, 0.0]
    
    spine_mid = (l_shoulder + r_shoulder) / 2
    hips_mid = (l_hip + r_hip) / 2
    spine_dir = spine_mid - hips_mid
    
    if np.linalg.norm(spine_dir) > 0.01:

        spine_quat = quaternion_desde_vectores(np.array([0, 1, 0]), normalize_vector(spine_dir), factor_suavizado=0.05)
        frame_data["mixamorig:Spine"]["rotation_quaternion"] = normalize_quaternion(spine_quat)
        frame_data["mixamorig:Spine1"]["rotation_quaternion"] = normalize_quaternion(spine_quat)
        frame_data["mixamorig:Spine2"]["rotation_quaternion"] = normalize_quaternion(spine_quat)
    
    # HOMBRO IZQUIERDO 
    if l_shoulder is not None and l_elbow is not None:
        left_arm_dir = l_elbow - l_shoulder
        if np.linalg.norm(left_arm_dir) > 0.01:
            # HOMBROS usan -Y (al revés que codos)
            left_arm_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(left_arm_dir), factor_suavizado=0.25)
            frame_data["mixamorig:LeftArm"]["rotation_quaternion"] = normalize_quaternion(left_arm_quat)
    
    # CODO IZQUIERDO
    if l_elbow is not None and l_wrist is not None:
        left_forearm_dir = l_wrist - l_elbow
        if np.linalg.norm(left_forearm_dir) > 0.01:
            # CODOS al 50% para que doblen más
            left_forearm_quat = quaternion_desde_vectores(np.array([0, 1, 0]), normalize_vector(left_forearm_dir), factor_suavizado=0.5)
            frame_data["mixamorig:LeftForeArm"]["rotation_quaternion"] = normalize_quaternion(left_forearm_quat)
    
    # MANO IZQUIERDA 
    
    # HOMBRO DERECHO
    if r_shoulder is not None and r_elbow is not None:
        right_arm_dir = r_elbow - r_shoulder
        if np.linalg.norm(right_arm_dir) > 0.01:
            # HOMBROS 
            right_arm_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(right_arm_dir), factor_suavizado=0.25)
            frame_data["mixamorig:RightArm"]["rotation_quaternion"] = normalize_quaternion(right_arm_quat)
    
    # CODO DERECHO - VECTOR 
    if r_elbow is not None and r_wrist is not None:
        right_forearm_dir = r_wrist - r_elbow
        if np.linalg.norm(right_forearm_dir) > 0.01:
            # CODOS al 50% para que doblen más
            right_forearm_quat = quaternion_desde_vectores(np.array([0, 1, 0]), normalize_vector(right_forearm_dir), factor_suavizado=0.5)
            frame_data["mixamorig:RightForeArm"]["rotation_quaternion"] = normalize_quaternion(right_forearm_quat)
    
    # MUSLO IZQUIERDO
    if l_hip is not None and l_knee is not None:
        left_upleg_dir = l_knee - l_hip
        if np.linalg.norm(left_upleg_dir) > 0.01:
            # Muslos con vector -Y 
            left_upleg_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(left_upleg_dir), factor_suavizado=0.5)
            frame_data["mixamorig:LeftUpLeg"]["rotation_quaternion"] = normalize_quaternion(left_upleg_quat)
    
    # RODILLA IZQUIERDA 
    if l_knee is not None and l_ankle is not None:
        left_leg_dir = l_ankle - l_knee
        if np.linalg.norm(left_leg_dir) > 0.01:
            # Rodillas con vector -Y 
            left_leg_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(left_leg_dir), factor_suavizado=0.5)
            frame_data["mixamorig:LeftLeg"]["rotation_quaternion"] = normalize_quaternion(left_leg_quat)
    
    # MUSLO DERECHO
    if r_hip is not None and r_knee is not None:
        right_upleg_dir = r_knee - r_hip
        if np.linalg.norm(right_upleg_dir) > 0.01:
            right_upleg_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(right_upleg_dir), factor_suavizado=0.5)
            frame_data["mixamorig:RightUpLeg"]["rotation_quaternion"] = normalize_quaternion(right_upleg_quat)
    
    # RODILLA DERECHA
    if r_knee is not None and r_ankle is not None:
        right_leg_dir = r_ankle - r_knee
        if np.linalg.norm(right_leg_dir) > 0.01:
            right_leg_quat = quaternion_desde_vectores(np.array([0, -1, 0]), normalize_vector(right_leg_dir), factor_suavizado=0.5)
            frame_data["mixamorig:RightLeg"]["rotation_quaternion"] = normalize_quaternion(right_leg_quat)
    
    # SUAVIZADO TEMPORAL 
    if previous_frame is not None:
        for bone_name in frame_data.keys():
            if bone_name in previous_frame:
                # Suavizar rotación
                current_rot = frame_data[bone_name]["rotation_quaternion"]
                previous_rot = previous_frame[bone_name]["rotation_quaternion"]
                frame_data[bone_name]["rotation_quaternion"] = smooth_quaternion_temporal(current_rot, previous_rot)
    
    return frame_data

# MAIN - CAPTURA DE VIDEO

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" No se pudo abrir la cámara")
        return
    
    frames_data = {}
    frame_count = 0
    recording = False
    previous_frame_data = None 
    
    print("\n" + "=" * 80)
    print(" CAPTURADOR MOTION CAPTURE - PIERNAS CORREGIDAS")

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Dibujar skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            if recording:
                frame_count += 1
                frame_data = capturar_frame(results.pose_landmarks, previous_frame_data)
                frames_data[str(frame_count)] = frame_data
                previous_frame_data = frame_data  
        
        # Mostrar estado - 1
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        status_text = " GRABANDO" if recording else " PAUSADO"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Piernas -Y CORREGIDAS", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Motion Capture - Metodo Mixamo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  
            break
        elif key == ord(' '): 
            recording = not recording
            if recording:
                print(f"  Grabación iniciada")
            else:
                print(f" Grabación pausada en frame {frame_count}")
        elif key == ord('r') or key == ord('R'):
            frames_data = {}
            frame_count = 0
            recording = False
            previous_frame_data = None
            print(" Grabación reiniciada")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar JSON
    if frames_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mocap_mixamo_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(frames_data, f, indent=4)
        
        print(f"\n Guardado: {filename}")
        print(f" Total de frames capturados: {frame_count}")
    else:
        print("\n No se capturaron frames")

if __name__ == "__main__":
    main()