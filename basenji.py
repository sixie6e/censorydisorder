import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile 

url = input("video URL: ")
temp_vid = "temp_vid.mp4"
temp_audio = "temp_audio.wav"
video_out = "video_out.mp4"
sample_rate = 22050 # standard

def yt_dlp(url: str, output_path: str) -> str | None:
    print("Getting audio...")
    
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_path,
        url
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"yt-dlp output: {result.stdout.strip()}")
        print(f"Audio downloaded: {output_path}")
        return output_path

def extract_features(audio_path: str) -> np.ndarray | None:
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


###########################################
# I need help with the comparison function.
def comparison():
	pass
###########################################


def mute(audio_path: str, detected_events: list[dict], event_to_target: str = 'dog_bark'):   
    print(f"\nProcessing audio: {event_to_target.title()})")

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
                        data[start_sample:end_sample] = 0.0")            
        sf.write(audio_path, data, sr)
        print(f"Audio saved: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"Error: {e}")
        return None

def replace(url: str, new_audio_path: str, video_out_path: str, temp_video_path: str) -> str | None:
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
        print("Not found.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        return None

def recombine(url: str):
    audio_file = yt_dlp(url, temp_audio)
    
    if not audio_file:
        return

    audio_duration = librosa.get_duration(path=audio_file)
    print(f"Audio duration: {audio_duration:.2f} seconds")
    features = extract_features(audio_file)
    
    if features is None:
        pass
        
    detected_events = comparison(features if features is not None else np.array([]), audio_duration)
    modified_audio_file = mute(audio_file, detected_events, event_to_target='dog_bark')
    
    if not modified_audio_file:
        if os.path.exists(temp_audio):
             os.remove(temp_audio)
        return
        
    final_video = replace(url, modified_audio_file, video_out, temp_vid)
    files_to_remove = [temp_audio, temp_vid]
    
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except OSError as e:
                print(f"Error: {e}")

recombine(url)
