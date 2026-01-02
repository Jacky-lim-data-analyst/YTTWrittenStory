from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript
import sys
from .storage import VideoMetadataDB


def show_video_details(transcript: FetchedTranscript):
    """Utility function to display basic transcript info"""
    print("-" * 10 + " Video details " + "-" * 10)
    print(f"Video ID: {transcript.video_id}")
    print(f"Language: {transcript.language}")
    print(f"Is the script auto-generated? {transcript.is_generated}\n")
    
    for idx, snippet in enumerate(transcript):
        if idx >= 10:
            break

        print(snippet.text)

def extract_video_details(transcript: FetchedTranscript) -> dict:
    return {
        "video_id": transcript.video_id,
        "language": transcript.language,
        "is_generated": transcript.is_generated
    }

def fetch_yt_transcript(
    video_id: str,
    title: str = ""
) -> tuple[list, dict]:
    if not title:
        title = "Unknown"

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id=video_id)

    raw_text = [snippet.text for snippet in fetched_transcript]

    db = VideoMetadataDB("video_metadata.db")
    db.create_table()

    metadata_dict = extract_video_details(fetched_transcript)
    if metadata_dict.get("video_id") is None:
        print("video_id is missing")
        sys.exit(1)

    # insert video metadata into db
    db.insert_video_metadata(
        (
            metadata_dict["video_id"],
            title,
            metadata_dict.get("language", ""),
            metadata_dict.get("is_generated", True)
        )
    )

    db.close()

    return raw_text, metadata_dict

# def main():
#     video_id = "k7qdSB_TYFE"
#     title = "The BIZARRE case of The Hammersmith Ghost"
#     ytt_api = YouTubeTranscriptApi()
#     fetched_transcript = ytt_api.fetch(video_id=video_id)

#     db = VideoMetadataDB("video_metadata.db")
#     db.create_table()

#     metadata_dict = extract_video_details(fetched_transcript)
#     if metadata_dict.get("video_id") is None:
#         print("video_id is missing")
#         sys.exit(1)

#     # insert video metadata into db
#     db.insert_video_metadata(
#         (
#             metadata_dict["video_id"],
#             title,
#             metadata_dict.get("language", ""),
#             metadata_dict.get("is_generated", True)
#         )
#     )

#     db.close()

# if __name__ == "__main__":
#     main()
