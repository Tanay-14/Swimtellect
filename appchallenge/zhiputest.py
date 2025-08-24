from dotenv import load_dotenv
load_dotenv()
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import zhipuai

# Load environment variables
client = zhipuai.ZhipuAI(api_key=os.getenv('BIGMODEL_API_KEY'))

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
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
    
    # Initialize MediaPipe Pose
    pose = mp.solutions.pose.Pose()
    image = cv2.imread(file_path)
    
    if image is None:
        return {
            'error': 'Could not load image',
            'landmarks': None,
            'analysis_data': None,
            'feedback': 'Image could not be loaded'
        }
    
    # Process image
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return {
            'error': 'No person detected',
            'landmarks': None,
            'analysis_data': None,
            'feedback': 'No person detected in the image'
        }
    
    # Extract all important landmarks
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape
    
    # Key landmarks for swimming analysis
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
    
    # Convert landmarks to pixel coordinates
    pixel_coords = {}
    for name, landmark in key_landmarks.items():
        pixel_coords[name] = {
            'x': landmark.x * w,
            'y': landmark.y * h,
            'z': landmark.z,
            'visibility': landmark.visibility
        }
    
    # Calculate important angles
    angles = {}
    
    # Left arm angle
    left_shoulder_coords = [pixel_coords['left_shoulder']['x'], pixel_coords['left_shoulder']['y']]
    left_elbow_coords = [pixel_coords['left_elbow']['x'], pixel_coords['left_elbow']['y']]
    left_wrist_coords = [pixel_coords['left_wrist']['x'], pixel_coords['left_wrist']['y']]
    angles['left_elbow'] = calculate_angle(left_shoulder_coords, left_elbow_coords, left_wrist_coords)
    
    # Right arm angle
    right_shoulder_coords = [pixel_coords['right_shoulder']['x'], pixel_coords['right_shoulder']['y']]
    right_elbow_coords = [pixel_coords['right_elbow']['x'], pixel_coords['right_elbow']['y']]
    right_wrist_coords = [pixel_coords['right_wrist']['x'], pixel_coords['right_wrist']['y']]
    angles['right_elbow'] = calculate_angle(right_shoulder_coords, right_elbow_coords, right_wrist_coords)
    
    # Body alignment analysis
    avg_shoulder_y = (key_landmarks['left_shoulder'].y + key_landmarks['right_shoulder'].y) / 2
    avg_hip_y = (key_landmarks['left_hip'].y + key_landmarks['right_hip'].y) / 2
    alignment_diff = abs(avg_shoulder_y - avg_hip_y)
    
    # Hand entry analysis
    left_hand_entry_diff = abs(pixel_coords['left_shoulder']['x'] - pixel_coords['left_wrist']['x'])
    right_hand_entry_diff = abs(pixel_coords['right_shoulder']['x'] - pixel_coords['right_wrist']['x'])
    
    # Posture analysis
    nose_y = key_landmarks['nose'].y
    left_shoulder_y = key_landmarks['left_shoulder'].y
    posture_good = nose_y > left_shoulder_y
    
    # Compile analysis data
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
    
    # Generate feedback
    feedback = []
    
    # Posture feedback
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
        'pixel_coords': pixel_coords,
        'analysis_data': analysis_data,
        'feedback': " ".join(feedback)
    }

def generate_zhipu_analysis(image_path, stroke_type="freestyle"):
    """Generate analysis using Zhipu AI with all the pose data"""
    
    # Get all analysis data
    analysis_result = analyze_image_with_variables(image_path)
    
    if analysis_result['error']:
        return analysis_result['feedback']
    
    # Prepare data for Zhipu AI
    analysis_data = analysis_result['analysis_data']
    landmarks = analysis_result['landmarks']
    pixel_coords = analysis_result['pixel_coords']
    
    # Create detailed prompt for Zhipu AI
    prompt = f"""
As a professional swimming coach, analyze this swimming technique data for {stroke_type} stroke:

TECHNICAL DATA:
- Left elbow angle: {analysis_data['angles']['left_elbow']:.1f} degrees
- Right elbow angle: {analysis_data['angles']['right_elbow']:.1f} degrees
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

Please provide:
1. Technical assessment of the stroke technique
2. Specific recommendations for improvement
3. Training tips for this specific stroke
4. Overall score (1-10) with explanation

Focus on {stroke_type} stroke technique specifically.
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

# Example usage
if __name__ == "__main__":
    # Test with a sample image
    test_image_path = "uploads/testimage.jpg"  # Update with your image path
    
    if os.path.exists(test_image_path):
        print("=== Swimming Analysis with Zhipu AI ===")
        print("\nGenerating detailed analysis...")
        
        # Get all variables from analyze_image function
        analysis_result = analyze_image_with_variables(test_image_path)
        
        if analysis_result['error']:
            print(f"Error: {analysis_result['error']}")
        else:
            print("\n=== Available Variables ===")
            print(f"Image dimensions: {analysis_result['analysis_data']['image_dimensions']}")
            print(f"Angles: {analysis_result['analysis_data']['angles']}")
            print(f"Alignment difference: {analysis_result['analysis_data']['alignment_diff']:.3f}")
            print(f"Hand entry differences: {analysis_result['analysis_data']['hand_entry_diffs']}")
            print(f"Posture good: {analysis_result['analysis_data']['posture_good']}")
            
            print("\n=== Basic Feedback ===")
            print(analysis_result['feedback'])
            
            print("\n=== Zhipu AI Analysis ===")
            ai_analysis = generate_zhipu_analysis(test_image_path, "freestyle")
            print(ai_analysis)
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please update the test_image_path variable with a valid image path.")