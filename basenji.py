import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Union, Optional

url = input("video URL: ")
temp_vid = "temp_vid.mp4"
temp_audio = "temp_audio.mp3"
video_out = "video_out.mp4"
sample_rate = 22050  # standard

def yt_dlp(url: str, output_path: str) -> Optional[str]:
    print("Getting audio...")
    
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", output_path,
        url
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"yt-dlp output: {result.stdout.strip()}")
        print(f"Audio downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_features(audio_path: str) -> Optional[np.ndarray]:
    print("\nExtracting MFCCs...")
    
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate) 
        if len(y) == 0:
            print("Audio empty.")
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) 
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        feature_vector = np.hstack([mfccs_mean, mfccs_std])
        return feature_vector
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def comparison(feature_vector: np.ndarray, duration_seconds: float) -> List[Dict]:
    print("Comparing...")
   
    # Placeholder for logic to compare feature_vector against a model
    detected_events = [
        {'event': 'dog_bark', 'start_time_sec': 0.2, 'end_time_sec': 1.3, 'confidence': 0.92},
    ]
    
    print("Events:")
    for event in detected_events:
        print(f"{event['event'].replace('_', ' ').title()}")
        print(f"{event['start_time_sec']:.2f} to {event['end_time_sec']:.2f}")
        print(f"Confidence Level: {event['confidence']:.2f}")

    return detected_events

def mute(audio_path: str, detected_events: List[Dict], event_to_target: str = 'dog_bark') -> Optional[str]:   
    print(f"\nProcessing audio: {event_to_target.title()}")

    try:
        data, sr = sf.read(audio_path, dtype='float32')       
        sec_to_indices = lambda t: int(t * sr)
        target_events = [e for e in detected_events if e['event'] == event_to_target]
        
        if not target_events:
            print("No events to process.")
            return audio_path

        for event in target_events:
            start_sample = sec_to_indices(event['start_time_sec'])
            end_sample = sec_to_indices(event['end_time_sec'])
            start_sample = max(0, start_sample)
            end_sample = min(len(data), end_sample)
            data[start_sample:end_sample] = 0.0           
        sf.write(audio_path, data, sr)
        print(f"Audio saved: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"Error: {e}")
        return None

def replace(url: str, new_audio_path: str, video_out_path: str, temp_video_path: str) -> Optional[str]:
    print("Downloading video without audio...")
    
    video_download_command = [
        "yt-dlp",
        "--format", "bestvideo",
        "-o", temp_video_path,
        url
    ]
    
    try:
        subprocess.run(video_download_command, check=True, capture_output=True, text=True)
        print(f"Video saved: {temp_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        return None

    combine_command = [
        "ffmpeg",
        "-i", temp_video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        video_out_path
    ]
    
    try:
        subprocess.run(combine_command, check=True, capture_output=True, text=True)
        print(f"Success. Video saved: {video_out_path}")
        return video_out_path
    except FileNotFoundError:
        print("FFmpeg not found. Please install ffmpeg.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        return None

def recombine(url: str):
    audio_file = yt_dlp(url, temp_audio)
    
    if not audio_file:
        return

    audio_duration = librosa.get_duration(filename=audio_file)
    print(f"Audio duration: {audio_duration:.2f} seconds")
    features = extract_features(audio_file)
    
    detected_events = comparison(features if features is not None else np.array([]), audio_duration)
    modified_audio_file = mute(audio_file, detected_events, event_to_target='dog_bark')

    if not modified_audio_file:
        if os.path.exists(temp_audio):
             os.remove(temp_audio)
        return
        
    final_video = replace(url, modified_audio_file, video_out, temp_vid)

    for f in [temp_audio, temp_vid]:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except OSError as e:
                print(f"Error deleting {f}: {e}")

if __name__ == "__main__":
    recombine(url)
