import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["sox_io"]
import os
import torch 
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import csv
import whisper
from jiwer import wer
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import random
import time

sample_rate=16000 #this is the output sample rate
project_root=Path(__file__).resolve().parent.parent #since data folder is outside python codes folder so we go to 2 parents back
testing_phase2_dir=project_root/"testing_attacks phase 2"
raw_directory=project_root/"small_data" 
clean_directory = testing_phase2_dir/"processed_small_data"
attack_directory=testing_phase2_dir/"indiv_attacked_processed_small_data"
accurate_csv_directory=raw_directory/"small_data.csv"
ground_truth=None

testing_phase2_dir.mkdir(parents=True, exist_ok=True)
clean_directory.mkdir(parents=True, exist_ok=True)
attack_directory.mkdir(parents=True, exist_ok=True)

#load speech detector (whisper model)
whisper_model=whisper.load_model("tiny") 
#load speaker detector 
speaker_model=EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir=str(project_root/"speaker_model"),
    local_strategy=LocalStrategy.COPY
)

#function that gets the accurate transcript of the audio
#exact transcripts are provided in a csv file named "small_data.csv"
def correct_transcript():
    df=pd.read_csv(accurate_csv_directory)
    ground_truth={}
    for index, row in df.iterrows():
        full_path=row["filename"] #this is now stored in form 'small_data/sample-000000.mp3'
        text=row["text"]    
        text=str(text).strip().lower()
        base=Path(full_path).name #we just get the 'sample-000000.mp3'
        ground_truth[base]=text
    return ground_truth

#function to get path of all audio mp3 files in the folder
def get_all_audio_files():
    audio_files=list(raw_directory.rglob("*.mp3"))
    print(f"Found {len(audio_files)} audio files.")
    return audio_files 

#function to load audio files using librosa
def load_audio(path, sample_r=sample_rate):
    y,x=librosa.load(path, sr=sample_r, mono=True)
    return y 

#function to normalize our audio recordings
def preprocess_audio(y): 
    max_val=np.max(np.abs(y))
    if max_val>0:
        y=y/max_val
    return y

#wrapper function: gets all paths, loads each audio and normalizes it 
def clean_files():
    files=get_all_audio_files()
    print(f"Processing {len(files)} files...")
    for i,f in enumerate(files, 1):
        y=load_audio(f)
        y_norm=preprocess_audio(y)
        out_name=f.with_suffix(".wav").name #changes file to wav
        out_path=clean_directory/out_name #use the preprocessed_data_small folder
        sf.write(out_path, y_norm, sample_rate) #saving the wav files in the preprocessed_data_small folder
        if i%10==0:
            print(f"[{i}/{len(files)}] cleaned") #to check all is going well 
    print("Done!")


#now we ll start defining manipulations functions
#function #1: we add white noise; REPLICATES play_white_noise()
def attack_white_noise(y):
    n=len(y)
    noise=np.random.uniform(-1, 1, size=n)    
    attacked=y + noise
    attacked=np.clip(attacked, -1.0, 1.0)
    return attacked

#function #2: we add a sine wave; REPLICATES play_sine_wave()
def attack_sine_wave(y):
    n=len(y)
    duration=n/sample_rate
    frequency=random.randint(200, 1000)
    t=np.linspace(0, duration, n, endpoint=False)
    tone=0.5*np.sin(2*np.pi*frequency*t)
    attacked=y+tone
    attacked = np.clip(attacked, -1.0, 1.0)
    return attacked

#function #3: add chirp; REPLICATES play_chirp() 
def attack_chirp(y):
    n=len(y)
    duration=n/sample_rate
    f_start=random.randint(200, 600)
    f_end=random.randint(1000, 3000)
    t=np.linspace(0, duration, n, endpoint=False)
    chirp=0.5*np.sin(2*np.pi*t*(f_start+(f_end-f_start)*t/duration))
    attacked=y+chirp
    attacked=np.clip(attacked, -1.0, 1.0)
    return attacked

#function #4: add glitch; REPLICATES play_glitch()
def attack_glitch(y):
    n=len(y)
    glitch=np.zeros(n)
    for i in range(0, n, 5000):
        burst=np.random.uniform(-1, 1, 500) 
        end_idx=min(i+500, n)
        glitch[i:end_idx]=burst[:end_idx-i]
    attacked=y+glitch
    attacked=np.clip(attacked, -1.0, 1.0)
    return attacked

#function #5: add overlapping sine waves; REPLICATES play_overlap()
def attack_overlap_sine(y):
    n=len(y)
    duration=n/sample_rate
    t=np.linspace(0, duration, n, endpoint=False)
    tone=np.zeros_like(t)
    for j in range(3):
        frequency=random.randint(200, 2000)
        tone+=0.3*np.sin(2*np.pi*frequency*t)
    tone=np.clip(tone, -1.0, 1.0)
    attacked=y+tone
    attacked=np.clip(attacked, -1.0, 1.0)
    return attacked


#we want to see the "difference" btwn both audios clean and attacked version so we ll measure SNR and l2 norm
#function calculates signal to noise ratio in db
def snr_db(clean, attacked):
    #for ex in time stretch we ll get diff array sizes so to ensure we get no errors we ll use an equal size to diff
    n=min(len(clean), len(attacked))
    clean=clean[:n]
    attacked=attacked[:n]
    noise=attacked-clean
    sig_power=np.mean(clean**2)
    noise_power=np.mean(noise**2) + 1e-12  #add small factor in case =0
    snr=10*np.log10(sig_power/noise_power)
    return snr

#function that transcribes audio using the loaded (prev) whisper model
def transcribe_audio(path):
    result=whisper_model.transcribe(str(path))
    text=result.get("text", "")
    return text.strip().lower()

#for each audio file we apply individual attacks and we ll measure the needed param
def apply_indiv_attacks():
    clean_files=list(clean_directory.glob("*.wav"))
    print(f"Found {len(clean_files)} clean files to attack.")
    #define which attacks to run + parameters
    attacks = {
        "white_noise": lambda y: attack_white_noise(y),
        "single_sine_wave": lambda y: attack_sine_wave(y),
        "chirp": lambda y: attack_chirp(y),
        "glitch": lambda y: attack_glitch(y),
        "overlap_sine_wave": lambda y: attack_overlap_sine(y),
    }
    metrics_excel=testing_phase2_dir/"indiv_attack_metrics.csv"
    with open(metrics_excel, "w", newline="", encoding="utf-8") as excelfile:
        writer=csv.writer(excelfile)
        writer.writerow(["file name", "type of attack", "true text", "clean transcript", "attacked transcript", "wer true vs clean", "wer true vs attack", "wer attack vs clean", "SNR (dB)"])
        for i, path in enumerate(clean_files, 1):
            y_clean,_=librosa.load(path, sr=sample_rate, mono=True)
            #we transcribe the clean audio
            clean_text=transcribe_audio(path)
            for attack_name, attack_fn in attacks.items():
                y_attack=attack_fn(y_clean)
                #subfolder for this attack
                attack_subdir=attack_directory/attack_name
                attack_subdir.mkdir(parents=True, exist_ok=True) 
                out_path=attack_subdir/path.name
                sf.write(out_path, y_attack, sample_rate)

                #we transcribe the attacked audio
                attack_text = transcribe_audio(out_path)
                snr_val=snr_db(y_clean, y_attack)
                mp3_name=path.with_suffix(".mp3").name #since our dict is stored using .mp3 names
                true_text=ground_truth.get(mp3_name)
                wer_clean=wer(true_text, clean_text)
                wer_attack=wer(true_text, attack_text)
                wer_ratio=wer(clean_text,attack_text)
                writer.writerow([path.name, attack_name, true_text, clean_text, attack_text, wer_clean, wer_attack, wer_ratio, snr_val])
                print(f"[{i}/{len(clean_files)}] files attacked")
    print("All individual attacks applied. Attacked audio saved in:", attack_directory)

#now we need to check speaker detection
#we ll get the embeddings of the clean and attacked files and compare them
#this function gets the embedding of a file
def speaker_embedding(path):
    y,_=librosa.load(str(path), sr=sample_rate, mono=True)
    signal=torch.FloatTensor(y).unsqueeze(0)
    embeddings=speaker_model.encode_batch(signal)
    embedding=embeddings.squeeze(0).detach().cpu().numpy() #so it saves as a vector
    return embedding 

#this function will calculate our cos similarity
def cosine_sim(y_clean, y_attack):
    y_clean=np.asarray(y_clean).flatten()
    y_attack=np.asarray(y_attack).flatten()
    denominator=(np.linalg.norm(y_clean))*(np.linalg.norm(y_attack))
    numerator=np.dot(y_clean, y_attack)
    cos=float(numerator/denominator)
    return cos

#check speaker detection for individual attacks
def speaker_recognition_indiv():
    clean_files=list(clean_directory.glob("*.wav"))
    attacks = {
        "white_noise": lambda y: attack_white_noise(y),
        "single_sine_wave": lambda y: attack_sine_wave(y),
        "chirp": lambda y: attack_chirp(y),
        "glitch": lambda y: attack_glitch(y),
        "overlap_sine_wave": lambda y: attack_overlap_sine(y),
    }
    speaker_metrics_excel=testing_phase2_dir /"indiv_attack_speaker_recognition.csv"
    with open(speaker_metrics_excel, "w", newline="", encoding="utf-8") as excelfile:
        writer=csv.writer(excelfile)
        writer.writerow(["file name", "type of attack", "speaker cosine"])
        for i, path in enumerate(clean_files, 1):
            clean_embedding=speaker_embedding(path)
            for attack_name in attacks:
                attack_path=attack_directory/attack_name/path.name
                attack_embedding=speaker_embedding(attack_path)
                cosine=cosine_sim(clean_embedding, attack_embedding)
                writer.writerow([path.name, attack_name, cosine])
                print(f"[{i}/{len(clean_files)}] indiv-attacks speaker done")
    print("Individual-Attacks Speaker Recognition is done")


def main():
    global ground_truth
    ground_truth=correct_transcript()
    clean_files()
    apply_indiv_attacks()
    speaker_recognition_indiv()
    
if __name__ == "__main__":
    main() 