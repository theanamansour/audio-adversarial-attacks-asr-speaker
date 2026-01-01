import numpy as np          
import sounddevice as sd    
import time                 
import random               

sample_rate=44100
duration=10               


#Function: White Noise
def play_white_noise():
    #Generate an array of random values between -1 and 1 for the whole duration
    noise=np.random.uniform(-1, 1, duration * sample_rate)
    sd.play(noise, sample_rate)  #Play the white noise
    sd.wait()                    #Wait until playback finishes


#Function: Sine Wave
def play_sine_wave():
    frequency = random.randint(200, 1000)  #Pick a random frequency between 200 Hz and 1000 Hz
    #Create an array of time points for the duration of the sound
    t=np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
    #Generate a sine wave based on the time array and frequency
    tone=0.5*np.sin(2*np.pi*frequency*t)
    sd.play(tone, sample_rate)  #Play the sine wave
    sd.wait()                   #Wait until playback finishes


#Function: Chirp
def play_chirp():
    f_start=random.randint(200, 600)     #Random starting frequency
    f_end=random.randint(1000, 3000)     #Random ending frequency
    t=np.linspace(0, duration, int(sample_rate * duration))  #Time array
    #Create a sine wave whose frequency increases over time from f_start to f_end
    chirp=0.5*np.sin(2*np.pi*t*(f_start+(f_end-f_start)*t/duration))
    sd.play(chirp, sample_rate)  #Play the chirp
    sd.wait()                    #Wait until playback finishes


#Function: Glitch Noise 
def play_glitch():
    noise=np.zeros(duration*sample_rate)  #Start with silence
    #Insert short bursts of random noise every ~0.1 seconds
    for i in range(0, len(noise), 5000):
        burst=np.random.uniform(-1, 1, 500) #Random burst of 500 samples
        end_idx=min(i+500, len(noise))      #Ensure we don’t go past array length
        noise[i:end_idx]=burst[:end_idx-i]  #Insert the burst into the noise array
    sd.play(noise, sample_rate)  #Play the glitch noise
    sd.wait()                    #Wait until playback finishes


#Function: Overlapping Sine Waves 
def play_overlap():
    t=np.linspace(0, duration, int(sample_rate*duration), endpoint=False)  #Time array
    tone=np.zeros_like(t)  #Initialize with silence
    for j in range(3):     #Add 3 sine waves of random frequencies
        frequency=random.randint(200, 2000)
        tone+=0.3*np.sin(2*np.pi*frequency*t)  #Sum of sine waves
    tone = np.clip(tone, -1, 1)  #Ensure values stay in the valid range for audio
    sd.play(tone, sample_rate)   #Play the combined tones
    sd.wait()                    #Wait until playback finishes


sounds=[play_white_noise, play_sine_wave, play_chirp, play_glitch, play_overlap]  #List of all sound functions

#Play 5 random sounds sequentially
for i in range(5):
    sound=random.choice(sounds)  #Randomly select a sound function
    print(f"Playing sound {i+1}: {sound.__name__}")  #Print which sound is being played
    sound()                         #Call the function to play the sound
    time.sleep(2)                   #Pause 2 seconds before playing the next sound
