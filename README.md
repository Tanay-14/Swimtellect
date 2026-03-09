# Swimtellect

Swimtellect is an AI-powered swimming analysis tool that evaluates swimming technique using computer vision and video analysis. The system processes swim footage to identify body position, stroke mechanics, and movement patterns, then provides feedback designed to help swimmers improve efficiency and form.

The project aims to make technical swim analysis more accessible by combining modern computer vision tools with AI-driven insights. Instead of relying solely on in-person coaching, swimmers can use video and automated analysis to better understand their performance.

## Features

**Pose and movement analysis**
Uses computer vision to detect body landmarks and analyze swimmer positioning throughout a stroke cycle.

**Stroke and timing analysis**
Tracks motion across video frames to estimate stroke rhythm, arm angles, and body alignment during swimming.

**Performance graphics**
Generates visualizations that highlight trends in stroke timing, consistency, and body positioning across a swim session.

**Training plan generator**
Creates suggested training plans based on detected technique issues and performance patterns.

**Tailored coaching videos**
Uses AI to recommend relevant YouTube coaching videos that address specific technique improvements identified in the analysis.

**Video-based analysis**
Supports recorded swim footage for technique evaluation.

## How It Works

A video of a swimmer is uploaded or captured through the web interface. The system processes the video frame by frame using pose estimation and computer vision. Body landmarks are tracked across time to analyze movement patterns such as arm recovery, stroke timing, and body alignment.

The extracted motion data is then evaluated to produce feedback, visual performance graphics, and training suggestions. The system can also generate targeted learning resources by recommending coaching videos that correspond to detected technique issues.

## Tech Stack

Python

Django

MediaPipe

OpenCV

Zhipu AI model (API-based and replaceable with other LLM providers)

HTML and CSS frontend

NumPy and related data processing libraries

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/swimtellect.git
cd swimtellect
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open a browser and navigate to:

```
http://localhost:5000
```

## License

This project is released under the MIT License.
