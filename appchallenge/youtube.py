import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("No YOUTUBE_API_KEY found in environment variables")

def get_youtube_videos(query, max_results=3):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response['items']:
        video = {
            'title': item['snippet']['title'],
            'thumbnail': item['snippet']['thumbnails']['medium']['url'],
            'videoId': item['id']['videoId']
        }
        videos.append(video)
    return videos