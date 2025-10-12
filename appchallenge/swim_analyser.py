import cv2
import os
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
import zhipuai

# Loads the environmental variables for Zhipu AI so that later functions can use the model
load_dotenv()
client = zhipuai.ZhipuAI(api_key=os.getenv('BIGMODEL_API_KEY'))

def analyze_media(file_path):
    if file_path.lower().endswith((".png",".jpg",".jpeg")):
        return analyze_image(file_path)
    elif file_path.lower().endswith((".mp4",".mov",".avi")):
        return analyze_video(file_path)
    return "file format not supported"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_image_with_variables(file_path):
    """Analyze image and return all important variables and data"""
    
    # Initializes the MediaPipe Pose model to deteck landmarks in the user's image
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image = cv2.imread(file_path)
    
    if image is None:
        return {
            'error': 'Could not load image',
            'annotated_image_path': None,
            'analysis_data': None,
            'feedback': 'Image could not be loaded'
        }
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return {
            'error': 'No person detected',
            'annotated_image_path': None,
            'analysis_data': None,
            'feedback': 'No person detected in the image'
        }

    # This function draws the landmarks on the image and saves it
    annotated_image = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    )


    path_parts = os.path.splitext(file_path)
    annotated_image_path = f"{path_parts[0]}_annotated.jpg"
    cv2.imwrite(annotated_image_path, annotated_image)
    
    # This part extracts the landmarks from the media pipe results
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape
    
    # Defines the key landmarks to be used in the analysis
    key_landmarks = {
        'nose': landmarks[mp.solutions.pose.PoseLandmark.NOSE],
        'left_shoulder': landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
        'right_shoulder': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
        'left_hip': landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP],
        'right_hip': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
        'left_elbow': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
        'right_elbow': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
        'left_wrist': landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
        'right_wrist': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
        'left_ankle': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
        'right_ankle': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE],
        'left_knee': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
        'right_knee': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
    }
    
    # Converts the landmarks to pixel coordinates to be sent to Zhipu AI
    pixel_coords = {}
    for name, landmark in key_landmarks.items():
        pixel_coords[name] = {
            'x': landmark.x * w,
            'y': landmark.y * h,
            'z': landmark.z,
            'visibility': landmark.visibility
        }
    
    angles = {}
    
    # Defines the Left arm angle
    left_shoulder_coords = [pixel_coords['left_shoulder']['x'], pixel_coords['left_shoulder']['y']]
    left_elbow_coords = [pixel_coords['left_elbow']['x'], pixel_coords['left_elbow']['y']]
    left_wrist_coords = [pixel_coords['left_wrist']['x'], pixel_coords['left_wrist']['y']]
    angles['left_elbow'] = calculate_angle(left_shoulder_coords, left_elbow_coords, left_wrist_coords)
    
    #Defines the Right arm angle
    right_shoulder_coords = [pixel_coords['right_shoulder']['x'], pixel_coords['right_shoulder']['y']]
    right_elbow_coords = [pixel_coords['right_elbow']['x'], pixel_coords['right_elbow']['y']]
    right_wrist_coords = [pixel_coords['right_wrist']['x'], pixel_coords['right_wrist']['y']]
    angles['right_elbow'] = calculate_angle(right_shoulder_coords, right_elbow_coords, right_wrist_coords)
    
    #Defines Left leg angle
    left_hip_coords = [pixel_coords['left_hip']['x'], pixel_coords['left_hip']['y']]
    left_knee_coords = [pixel_coords['left_knee']['x'], pixel_coords['left_knee']['y']]
    left_ankle_coords = [pixel_coords['left_ankle']['x'], pixel_coords['left_ankle']['y']]
    angles['left_knee'] = calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)
    
    #Defines Right leg angle
    right_hip_coords = [pixel_coords['right_hip']['x'], pixel_coords['right_hip']['y']]
    right_knee_coords = [pixel_coords['right_knee']['x'], pixel_coords['right_knee']['y']]
    right_ankle_coords = [pixel_coords['right_ankle']['x'], pixel_coords['right_ankle']['y']]
    angles['right_knee'] = calculate_angle(right_hip_coords, right_knee_coords, right_ankle_coords)
    
    # Calculates Body alignment analysis
    avg_shoulder_y = (key_landmarks['left_shoulder'].y + key_landmarks['right_shoulder'].y) / 2
    avg_hip_y = (key_landmarks['left_hip'].y + key_landmarks['right_hip'].y) / 2
    alignment_diff = abs(avg_shoulder_y - avg_hip_y)
    
    #Calculates Hand entry analysis
    left_hand_entry_diff = abs(pixel_coords['left_shoulder']['x'] - pixel_coords['left_wrist']['x'])
    right_hand_entry_diff = abs(pixel_coords['right_shoulder']['x'] - pixel_coords['right_wrist']['x'])
    
    #Calculates Posture analysis
    nose_y = key_landmarks['nose'].y
    left_shoulder_y = key_landmarks['left_shoulder'].y
    posture_good = nose_y > left_shoulder_y
    
    # Compiles the analysis data
    analysis_data = {
        'image_dimensions': {'width': w, 'height': h},
        'angles': angles,
        'alignment_diff': alignment_diff,
        'hand_entry_diffs': {
            'left': left_hand_entry_diff,
            'right': right_hand_entry_diff
        },
        'posture_good': posture_good,
        'landmark_visibility': {name: coords['visibility'] for name, coords in pixel_coords.items()}
    }
    
    feedback = []
    
    if posture_good:
        feedback.append("Posture is right.")
    else:
        feedback.append("Posture is not right.")
    
    # Body alignment feedback
    required_landmarks = [key_landmarks['left_shoulder'], key_landmarks['right_shoulder'], 
                         key_landmarks['left_hip'], key_landmarks['right_hip']]
    if any(lm.visibility < 0.5 for lm in required_landmarks):
        feedback.append("Cannot assess body alignment: full body not visible in the image.")
    else:
        if alignment_diff < 0.05:
            feedback.append("Body alignment is good (body is straight).")
        else:
            feedback.append("Body alignment could be improved (body is not straight).")
    
    # Arm angle feedback
    for side in ['left', 'right']:
        angle = angles[f'{side}_elbow']
        feedback.append(f"{side.capitalize()} elbow angle: {angle:.1f} degrees.")
        if angle > 160:
            feedback.append(f"{side.capitalize()} arm is well extended.")
        elif angle < 100:
            feedback.append(f"{side.capitalize()} arm is too bent.")
        else:
            feedback.append(f"{side.capitalize()} arm is moderately bent.")
    
    # Hand entry feedback
    threshold = 40  # pixels
    for side in ['left', 'right']:
        diff = analysis_data['hand_entry_diffs'][side]
        if diff < threshold:
            feedback.append(f"{side.capitalize()} hand entry is good (in line with shoulder).")
        else:
            feedback.append(f"{side.capitalize()} hand entry could be improved (not in line with shoulder).")
    
    return {
        'error': None,
        'landmarks': key_landmarks,
        'annotated_image_path': annotated_image_path,
        'pixel_coords': pixel_coords,
        'analysis_data': analysis_data,
        'feedback': " ".join(feedback)
    }

def generate_zhipu_analysis(image_path, stroke_type="freestyle", swimming_level="beginner"):
    """Generate analysis using Zhipu AI with all the pose data"""
    
    # Gain all analysis data
    analysis_result = analyze_image_with_variables(image_path)
    
    if analysis_result['error']:
        return analysis_result['feedback']
    
    # This prepares the data for Zhipu AI
    analysis_data = analysis_result['analysis_data']
    landmarks = analysis_result['landmarks']
    pixel_coords = analysis_result['pixel_coords']
    
    # Prompt for Zhipu AI
    prompt = f"""
As a professional swimming coach, analyze this swimming technique data for {stroke_type} stroke in english in second person. The swimmer's skill level is: {swimming_level.upper()}.

TECHNICAL DATA:
- Left elbow angle: {analysis_data['angles']['left_elbow']:.1f} degrees
- Right elbow angle: {analysis_data['angles']['right_elbow']:.1f} degrees
- Left knee angle: {analysis_data['angles']['left_knee']:.1f} degrees
- Right knee angle: {analysis_data['angles']['right_knee']:.1f} degrees
- Body alignment difference: {analysis_data['alignment_diff']:.3f}
- Left hand entry distance from shoulder: {analysis_data['hand_entry_diffs']['left']:.1f} pixels
- Right hand entry distance from shoulder: {analysis_data['hand_entry_diffs']['right']:.1f} pixels
- Posture assessment: {'Good' if analysis_data['posture_good'] else 'Needs improvement'}

LANDMARK POSITIONS (normalized coordinates):
- Nose: ({landmarks['nose'].x:.3f}, {landmarks['nose'].y:.3f})
- Left shoulder: ({landmarks['left_shoulder'].x:.3f}, {landmarks['left_shoulder'].y:.3f})
- Right shoulder: ({landmarks['right_shoulder'].x:.3f}, {landmarks['right_shoulder'].y:.3f})
- Left hip: ({landmarks['left_hip'].x:.3f}, {landmarks['left_hip'].y:.3f})
- Right hip: ({landmarks['right_hip'].x:.3f}, {landmarks['right_hip'].y:.3f})
- Left knee: ({landmarks['left_knee'].x:.3f}, {landmarks['left_knee'].y:.3f})
- Right knee: ({landmarks['right_knee'].x:.3f}, {landmarks['right_knee'].y:.3f})
- Left ankle: ({landmarks['left_ankle'].x:.3f}, {landmarks['left_ankle'].y:.3f})
- Right ankle: ({landmarks['right_ankle'].x:.3f}, {landmarks['right_ankle'].y:.3f})

IMPORTANT ANALYSIS NOTES:
If knee_y and ankle_y are greater than hip_y in freestyle, then legs are sinking below torso → drag increases. Proper technique: legs near surface.
If nose_y is greater than shoulder_y in freestyle, then head is lifted → hips drop. Proper technique: head stays low and aligned with spine.
If elbow_y < wrist_y during recovery in freestyle, then arm recovery is flat. Proper technique: elbow should recover higher than wrist.
If wrist_x does not move inward toward shoulder_x during pull in freestyle, then catch is weak. Proper technique: hand should press slightly inward.

If hip_y > shoulder_y in backstroke, then hips are dropping → breaking streamline. Proper technique: hips close to surface.
If nose_y varies largely between frames in backstroke, then head is unstable. Proper technique: head stable, water at ear level.
If wrist_x at entry is far outside or inside shoulder_x in backstroke, then arm entry is inefficient. Proper technique: hand enters above shoulder, pinky first.
If knee_y > hip_y in backstroke, then knees break surface → drag. Proper technique: kick from hips, minimal knee bend.

If ankle_x left and right are not mirrored in breaststroke, then kick is asymmetrical. Proper technique: both legs move symmetrically.
If knee_y is close to hip_y or torso in breaststroke, then knees pulled too far forward → drag. Proper technique: knees bend moderately, stay behind hips.
If elbow_y > wrist_y during recovery in breaststroke, then arms are lifting upward instead of forward. Proper technique: arms recover forward under surface.
If distance between ankle_y and hip_y stays large after kick in breaststroke, then legs not fully extended. Proper technique: legs extend fully during glide.

If chest_y and hip_y move simultaneously in butterfly, then undulation is flat. Proper technique: chest presses down first, hips follow in wave.
If elbow_y < wrist_y during recovery in butterfly, then arms are dragging through water. Proper technique: elbows higher than wrists.
If ankle_y does not reach lowest point during pull in butterfly, then kick timing is off. Proper technique: downkick should align with pull.
If wrist_y left and right are not equal at entry in butterfly, then arms are entering unevenly. Proper technique: both arms enter together at shoulder width.

SWIMMING LEVEL CONTEXT:
- BEGINNER: Focus on basic technique, body position, and breathing. Keep feedback simple and encouraging.
- INTERMEDIATE: Address more technical aspects, stroke efficiency, and power generation.
- ADVANCED: Focus on fine-tuning technique, race-specific improvements, and advanced training concepts.
- COMPETITIVE: Emphasize race performance, advanced technique optimization, and elite-level training.

Please only these sections and provide in humanlike language in english in second person:

First, provide a JSON object with scores from 1 to 10 and give higher scores to more experienced swimmers. Give those scores for the following categories based on the data. The JSON object should be enclosed in triple backticks.
```json
{{
  "scores": {{ "Coordination": <score>, "Breathing technique": <score>, "Body alignment": <score>, "Arm stroke efficiency": <score>, "Kick technique": <score> }}
}}
```

1. Technical assessment of the stroke technique (tailored to {swimming_level} level)
2. Specific recommendations for improvement (appropriate for {swimming_level} level)
3. Training tips and specific drills for {stroke_type} that would help the swimmer improve their technique (suitable for {swimming_level} level)
4. Dry excersizes that would help the swimmer improve their technique (suitable for {swimming_level} level)

... Only include the four requested sections exactly as listed. Do not add any other sections or headings.
Focus on {stroke_type} stroke technique specifically and provide feedback appropriate for a {swimming_level} level swimmer.
"""
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def generate_video_zhipu_analysis(video_path, stroke_type="freestyle", swimming_level="beginner"):
    """Generate AI analysis for video using frame-by-frame data"""
    
    video_analysis_result = analyze_video(video_path)
    
    if "Error" in video_analysis_result:
        return video_analysis_result
    
    # Extracting important frames for analysis which is why the it takes longer to analyze a video
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error: Could not open video file for AI analysis"
    
    # Get the video properties using computer vision
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total_frames: {total_frames}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"duration: {duration}")
    
    # Analyze about 3-5 key frames for AI analysis depending on the length of the video
    key_frames = []
    frame_positions = [0.25, 0.5, 0.75]  # Analyze at 25%, 50%, and 75% of video
    
    for position in frame_positions:
        frame_number = int(total_frames * position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        print(f"Frame {frame_number} read: {ret}")
        
        if ret:
            # Save the frame temporarily with high quality 
            temp_frame_path = f"ai_frame_{position}.jpg"
            
            # Ensure frame is in BGR format because OpenCV only works with BGR: Issue that I faced
            if frame is not None and frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Save it with high quality compression
                success = cv2.imwrite(temp_frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"AI frame saved successfully: {success}")
                
                if success:
                    frame_analysis = analyze_image_with_variables(temp_frame_path)
                    
                    if frame_analysis['error'] is None:
                        key_frames.append({
                            'position': position,
                            'time_seconds': frame_number / fps,
                            'analysis': frame_analysis
                        })
                    else:
                        print(f"AI frame analysis error: {frame_analysis['error']}")
                else:
                    print(f"Failed to save AI frame at position {position}")
            
            # Clean all the frames so it doesn't take up space
            import os
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
    
    cap.release()
    
    if not key_frames:
        return "Error: No valid frames could be analyzed for AI assessment"
    
    # Prepare its data for AI analysis
    first_frame = key_frames[0]['analysis']
    middle_frame = key_frames[len(key_frames)//2]['analysis']
    last_frame = key_frames[-1]['analysis']
    
    #Prompt for Zhipu AI
    prompt = f"""
As a professional swimming coach, analyze this swimming video data for {stroke_type} stroke in english in second person. The swimmer's skill level is: {swimming_level.upper()}.:

VIDEO INFORMATION:
- Video duration: {duration:.1f} seconds
- Frames analyzed: {len(key_frames)} key frames
- Stroke type: {stroke_type}

TECHNICAL DATA (averaged across key frames):
- Left elbow angles: {first_frame['analysis_data']['angles']['left_elbow']:.1f}°, {middle_frame['analysis_data']['angles']['left_elbow']:.1f}°, {last_frame['analysis_data']['angles']['left_elbow']:.1f}°
- Right elbow angles: {first_frame['analysis_data']['angles']['right_elbow']:.1f}°, {middle_frame['analysis_data']['angles']['right_elbow']:.1f}°, {last_frame['analysis_data']['angles']['right_elbow']:.1f}°
- Left knee angles: {first_frame['analysis_data']['angles']['left_knee']:.1f}°, {middle_frame['analysis_data']['angles']['left_knee']:.1f}°, {last_frame['analysis_data']['angles']['left_knee']:.1f}°
- Right knee angles: {first_frame['analysis_data']['angles']['right_knee']:.1f}°, {middle_frame['analysis_data']['angles']['right_knee']:.1f}°, {last_frame['analysis_data']['angles']['right_knee']:.1f}°

KEY FRAME LANDMARK POSITIONS:
Frame 1 (25% of video):
- Left knee: ({first_frame['landmarks']['left_knee'].x:.3f}, {first_frame['landmarks']['left_knee'].y:.3f})
- Left ankle: ({first_frame['landmarks']['left_ankle'].x:.3f}, {first_frame['landmarks']['left_ankle'].y:.3f})

Frame 2 (50% of video):
- Left knee: ({middle_frame['landmarks']['left_knee'].x:.3f}, {middle_frame['landmarks']['left_knee'].y:.3f})
- Left ankle: ({middle_frame['landmarks']['left_ankle'].x:.3f}, {middle_frame['landmarks']['left_ankle'].y:.3f})

Frame 3 (75% of video):
- Left knee: ({last_frame['landmarks']['left_knee'].x:.3f}, {last_frame['landmarks']['left_knee'].y:.3f})
- Left ankle: ({last_frame['landmarks']['left_ankle'].x:.3f}, {last_frame['landmarks']['left_ankle'].y:.3f})

IMPORTANT VIDEO ANALYSIS NOTES:
If knee_y and ankle_y are greater than hip_y in freestyle, then legs are sinking below torso → drag increases. Proper technique: legs near surface.
If nose_y is greater than shoulder_y in freestyle, then head is lifted → hips drop. Proper technique: head stays low and aligned with spine.
If elbow_y < wrist_y during recovery in freestyle, then arm recovery is flat. Proper technique: elbow should recover higher than wrist.
If wrist_x does not move inward toward shoulder_x during pull in freestyle, then catch is weak. Proper technique: hand should press slightly inward.

If hip_y > shoulder_y in backstroke, then hips are dropping → breaking streamline. Proper technique: hips close to surface.
If nose_y varies largely between frames in backstroke, then head is unstable. Proper technique: head stable, water at ear level.
If wrist_x at entry is far outside or inside shoulder_x in backstroke, then arm entry is inefficient. Proper technique: hand enters above shoulder, pinky first.
If knee_y > hip_y in backstroke, then knees break surface → drag. Proper technique: kick from hips, minimal knee bend.

If ankle_x left and right are not mirrored in breaststroke, then kick is asymmetrical. Proper technique: both legs move symmetrically.
If knee_y is close to hip_y or torso in breaststroke, then knees pulled too far forward → drag. Proper technique: knees bend moderately, stay behind hips.
If elbow_y > wrist_y during recovery in breaststroke, then arms are lifting upward instead of forward. Proper technique: arms recover forward under surface.
If distance between ankle_y and hip_y stays large after kick in breaststroke, then legs not fully extended. Proper technique: legs extend fully during glide.

If chest_y and hip_y move simultaneously in butterfly, then undulation is flat. Proper technique: chest presses down first, hips follow in wave.
If elbow_y < wrist_y during recovery in butterfly, then arms are dragging through water. Proper technique: elbows higher than wrists.
If ankle_y does not reach lowest point during pull in butterfly, then kick timing is off. Proper technique: downkick should align with pull.
If wrist_y left and right are not equal at entry in butterfly, then arms are entering unevenly. Proper technique: both arms enter together at shoulder width.
- Look for consistency in technique throughout the video duration.
- Assess if the swimmer maintains proper form or if technique degrades over time.

Please provide in humanlike language in english in second person:

First, provide a JSON object with scores from 1 to 10 for the following categories based on the data. The JSON object should be enclosed in triple backticks.
```json
{{
  "scores": {{ "Coordination": <score>, "Breathing technique": <score>, "Body alignment": <score>, "Arm stroke efficiency": <score>, "Kick technique": <score> }}
}}
```

1. Technical assessment of the stroke technique throughout the video
2. Specific recommendations for improvement based on video analysis
3. Training tips and specific drills for this specific stroke that would help the swimmer improve their technique
4. Dry excersizes that would help the swimmer improve their technique (suitable for {swimming_level} level)
Focus on {stroke_type} stroke technique specifically and how it changes throughout the video and provide feedback appropriate for a {swimming_level} level swimmer.
"""
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def analyze_image(file_path):
    pose = mp.solutions.pose.Pose()
    image = cv2.imread(file_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return "No person detected in the image"
    nose = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    left_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

    h, w, _ = image.shape
    shoulder_coords = [left_shoulder.x * w, left_shoulder.y * h]
    elbow_coords = [left_elbow.x * w, left_elbow.y * h]
    wrist_coords = [left_wrist.x * w, left_wrist.y * h]

    elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)

    feedback = []
    if nose.y > left_shoulder.y:
        feedback.append("Posture is right.")
    else:
        feedback.append("Posture is not right.")

    required_landmarks = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(lm.visibility < 0.5 for lm in required_landmarks):
        feedback.append("Cannot assess body alignment: full body not visible in the image.")
    else:
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        alignment_diff = abs(avg_shoulder_y - avg_hip_y)
        if alignment_diff < 0.05:
            feedback.append("Body alignment is good (body is straight).")
        else:
            feedback.append("Body alignment could be improved (body is not straight).")

    feedback.append(f"Left elbow angle: {elbow_angle:.1f} degrees.")
    if elbow_angle > 160:
        feedback.append("Arm is well extended.")
    elif elbow_angle < 100:
        feedback.append("Arm is too bent.")
    else:
        feedback.append("Arm is moderately bent.")

    left_shoulder_x = left_shoulder.x * w
    left_wrist_x = left_wrist.x * w
    right_shoulder_x = right_shoulder.x * w
    right_wrist_x = right_wrist.x * w

    left_hand_entry_diff = abs(left_shoulder_x - left_wrist_x)
    right_hand_entry_diff = abs(right_shoulder_x - right_wrist_x)

    # Threshold
    threshold = 40 

    if left_hand_entry_diff < threshold:
        feedback.append("Left hand entry is good (in line with shoulder).")
    else:
        feedback.append("Left hand entry could be improved (not in line with shoulder).")

    if right_hand_entry_diff < threshold:
        feedback.append("Right hand entry is good (in line with shoulder).")
    else:
        feedback.append("Right hand entry could be improved (not in line with shoulder).")

    return " ".join(feedback)

def analyze_video(file_path):
    """Analyze video by extracting frames and performing pose analysis"""
    import cv2
    
    
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        return "Error: Could not open video file"
    
    # Get the properties of the video 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Extract frames at regular intervals; the interval can be increased for greater accuracy but takes longer
    frame_interval = 15
    analyzed_frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
       
        if not ret:
            break
            
                
        if frame_count % frame_interval == 0:
            print("condition is correct")
            # Saves frame temporarily with high quality
            temp_frame_path = f"temp_frame_{frame_count}.jpg"
            
            if frame is not None and frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                success = cv2.imwrite(temp_frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"Frame saved successfully: {success}")
                
                if success:
                    # Analyze the frame
                    frame_analysis = analyze_image_with_variables(temp_frame_path)
                    
                    if frame_analysis['error'] is None:
                        analyzed_frames.append({
                            'frame_number': frame_count,
                            'time_seconds': frame_count / fps,
                            'analysis': frame_analysis
                        })
                    else:
                        print(f"Frame analysis error: {frame_analysis['error']}")
                else:
                    print(f"Failed to save frame {frame_count}")
            
            # Clean up all the frames so it doesn't take up space
            import os
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        frame_count += 1
    
    cap.release()
    
    if not analyzed_frames:
        return "Error: No valid frames could be analyzed from the video"
    
    # create the overall video analysis
    return compile_video_analysis(analyzed_frames, duration, fps)

def compile_video_analysis(analyzed_frames, duration, fps):
    """Compile analysis from multiple frames into overall video assessment"""
    
    if not analyzed_frames:
        return "No frames were successfully analyzed"
    
    # Calculate averages across all frames
    total_elbow_angles = {'left': [], 'right': []}
    total_knee_angles = {'left': [], 'right': []}
    total_alignment_diffs = []
    total_hand_entry_diffs = {'left': [], 'right': []}
    posture_scores = []
    
    for frame_data in analyzed_frames:
        analysis = frame_data['analysis']
        analysis_data = analysis['analysis_data']
        
        # Collect elbow angles
        total_elbow_angles['left'].append(analysis_data['angles']['left_elbow'])
        total_elbow_angles['right'].append(analysis_data['angles']['right_elbow'])
        
        # Collect knee angles
        total_knee_angles['left'].append(analysis_data['angles']['left_knee'])
        total_knee_angles['right'].append(analysis_data['angles']['right_knee'])
        
        # Collect alignment differences
        total_alignment_diffs.append(analysis_data['alignment_diff'])
        
        # Collect hand entry differences
        total_hand_entry_diffs['left'].append(analysis_data['hand_entry_diffs']['left'])
        total_hand_entry_diffs['right'].append(analysis_data['hand_entry_diffs']['right'])
        
        # Collect posture scores
        posture_scores.append(1 if analysis_data['posture_good'] else 0)
    
    # Calculate averages
    avg_analysis = {
        'left_elbow_angle': sum(total_elbow_angles['left']) / len(total_elbow_angles['left']),
        'right_elbow_angle': sum(total_elbow_angles['right']) / len(total_elbow_angles['right']),
        'left_knee_angle': sum(total_knee_angles['left']) / len(total_knee_angles['left']),
        'right_knee_angle': sum(total_knee_angles['right']) / len(total_knee_angles['right']),
        'alignment_diff': sum(total_alignment_diffs) / len(total_alignment_diffs),
        'left_hand_entry_diff': sum(total_hand_entry_diffs['left']) / len(total_hand_entry_diffs['left']),
        'right_hand_entry_diff': sum(total_hand_entry_diffs['right']) / len(total_hand_entry_diffs['right']),
        'posture_score': sum(posture_scores) / len(posture_scores),
        'frames_analyzed': len(analyzed_frames),
        'video_duration': duration
    }
    
    
    feedback = []
    feedback.append(f"Video Analysis Summary:")
    feedback.append(f"- Duration: {duration:.1f} seconds")
    feedback.append(f"- Frames analyzed: {len(analyzed_frames)}")
    feedback.append(f"- Average left elbow angle: {avg_analysis['left_elbow_angle']:.1f}°")
    feedback.append(f"- Average right elbow angle: {avg_analysis['right_elbow_angle']:.1f}°")
    feedback.append(f"- Average left knee angle: {avg_analysis['left_knee_angle']:.1f}°")
    feedback.append(f"- Average right knee angle: {avg_analysis['right_knee_angle']:.1f}°")
    
    if avg_analysis['posture_score'] > 0.8:
        feedback.append("- Posture: Consistently good throughout the video")
    elif avg_analysis['posture_score'] > 0.5:
        feedback.append("- Posture: Generally good with some inconsistencies")
    else:
        feedback.append("- Posture: Needs improvement - inconsistent throughout the video")
    
    if avg_analysis['alignment_diff'] < 0.05:
        feedback.append("- Body alignment: Consistently straight")
    else:
        feedback.append("- Body alignment: Could be improved - some misalignment detected")
    
    return " ".join(feedback)
#This function generates a personalized training plan based on your past uploads
def generate_training_plan(swimming_level, uploads):
    """Generate a training plan using Zhipu AI based on user's level and upload history."""

    if not uploads:
        return "Not enough data to generate a training plan. Please upload some media for analysis first."

    analysis_summaries = []
    for upload in uploads[:5]:
        if upload.analysis_summary:
            summary_preview = (upload.analysis_summary[:250] + '...') if len(upload.analysis_summary) > 250 else upload.analysis_summary
            analysis_summaries.append(
                f"- For a {upload.get_stroke_display()} swim on {upload.uploaded_at.strftime('%Y-%m-%d')}, the analysis was: {summary_preview}"
            )

    if not analysis_summaries:
        return "No analyses found in your upload history. Cannot generate a training plan."

    history_summary = "\n".join(analysis_summaries)

    prompt = f"""
As a world-class swimming coach, create a personalized swimming training set for a swimmer with the following profile. The response should be in English and use the second person ("you").

SWIMMER PROFILE:
- Skill Level: {swimming_level.upper()}
- Summary of Recent Analysis History:
{history_summary}

Based on this history, identify the swimmer's main weaknesses and strengths. Then, create a detailed one-week training plan to help them improve. The plan should be structured with clear daily workouts.

Please provide the training plan in a human-like, encouraging tone. Use Markdown for formatting. Structure your response with the following sections:

 Your Personalized 1-Week Training Plan
  Primary Focus Areas:
    [List 2-3 key areas for improvement based on the analysis, e.g., High Elbow Catch, Body Rotation]

  Weekly Workout Schedule:  
      Day 1: [Focus of the day, e.g., Technique & Form]
          Warm-up:   [e.g., 200m easy swim, 4x50m drills]
          Main Set:   [e.g., 8x100m freestyle with focus on high elbow, 30s rest]
          Cool-down:   [e.g., 100m easy backstroke]
      Day 2: [Focus of the day]**
        ... (continue for a few more days, suggesting rest days as appropriate)

Keep the advice practical and tailored to the swimmer's {swimming_level} level.
"""

    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI training plan: {str(e)}"
