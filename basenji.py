import pytube
import librosa
import numpy as np
import os


url = raw_input("video URL: ")
audio = "temp_audio.wav"
sample_rate = 22050 

def download_audio(url: str, output_path: str) -> str:
    print(f"downloading audio from {url}...")
    try:
        yt = pytube.YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        downloaded_file = audio_stream.download(filename=output_path)
        print(f"audio downloaded to: {downloaded_file}")
        
        if not downloaded_file.endswith(".wav"):
            base, ext = os.path.splitext(downloaded_file)
            new_file = base + ".wav"
            os.rename(downloaded_file, new_file)
            print(f"renamed to {new_file}")  # when renaming extention fails, try ffmpeg/pydub
            return new_file

        return downloaded_file
    except Exception as e:
        print(f"error: {e}")
        return None

def extract_features(audio_path: str) -> np.ndarray:
	# I had help here, grabs audio and extracts Mel-Frequency Cepstral Coefficients
    print("extracting audio features (MFCCs)...")
    try:
        # load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)        
        # extract mfcc's
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)        
        #normalize(mean and std deviation of frames)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)        
        # simple feature vetcor, mean and standard deviation of MFCCs
        feature_vector = np.hstack([mfccs_mean, mfccs_std])
        return feature_vector
    except Exception as e:
        print(f"an error occurred during feature extraction: {e}")
        return None

def compare_to_database(feature_vector: np.ndarray) -> dict:
    """
    placeholder for the machine learning classification step which would probably involve:
    1. loading a pre-trained machine learning model
    2. loading database of known gunshot and dog bark features('samples').
    3. classify the feature_vector and return a confidence score.
    """
    # assume the video/song is 60 seconds long
    mock_events = {
        'gunshot': {'time_seconds': 15.2, 'confidence': 0.89},
        'dog_bark': {'time_seconds': 42.1, 'confidence': 0.95}
    }
    # compare for 'gunshot' and 'dog_bark' using Euclidean distance, lower = higher 'confidence'.    
    return mock_events


def analyze_audio(url: str):
    print("--- starting analysis ---")
    audio_file = download_audio(url, audio)
    
    if not audio_file:
        return   
         
    features = extract_features(audio_file)
    if features is None:
        os.remove(audio_file)
        return
         
    print("comparing features against database...")
    filtered_events = compare_to_database(features)
    
    if filtered_events:
        print("\n--- detected Events ---")
        for event, data in filtered_events.items():
            print(f"**{event.replace('_', ' ').title()}** detected!")
            print(f"   - time: **{data['time_seconds']} seconds**")
            print(f"   - confidence: **{data['confidence']:.2f}**")
        print("-----------------------")
    else:
        print("no matching events found.")
        
    try:
        os.remove(audio_file)
        print(f"\ncleaned up temporary file: {audio_file}")
    except OSError as e:
        print(f"error removing temporary file: {e}")


analyze_audio(url)
