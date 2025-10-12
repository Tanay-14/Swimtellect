import os
from googleapiclient.discovery import build
YOUTUBE_API_KEY = "AIzaSyCYcRBormMJhQK_ciFMWWUo9RTDfJlyz88"

# It's recommended to load the API key from environment variables for security
# You will need to set this environment variable in your development and production environments.
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyCYcRBormMJhQK_ciFMWWUo9RTDfJlyz88")

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