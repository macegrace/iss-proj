#   Martin Zaťovič (xzatov00)
#   ISS - projekt, implementácia spracovania signálov
#   školský rok 2020/2021

import numpy
import librosa
from matplotlib import pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from pydub import AudioSegment
import pandas as pd
import statsmodels.api as sm
import math
import cmath
import scipy

def hamming_window(array):
    arr = numpy.copy(array)
    hamming = numpy.hamming(len(arr))
    
    #plt.figure(figsize=(20, 5))   
    #plt.plot(hamming)
    #plt.title("Hamming window")
    #plt.ylabel("Amplitude")
    #plt.xlabel("Sample")
    #plt.show()
    
    #plt.figure(figsize=(20,5))
    #A = numpy.fft.fft(hamming, 2048) / 1024
    #mag = numpy.abs(numpy.fft.fftshift(A))
    #freq = numpy.linspace(-0.5, 0.5, len(A))
    #response = 20 * numpy.log10(mag)
    #response = numpy.clip(response, -100, 100)
    #plt.plot(freq, response)
    #plt.title("Graf Hammingovho okna v spektrálnej oblasti")
    #plt.ylabel("Magnitúda [dB]")
    #plt.xlabel("Normalizovaná frekvencia [cykly za sekundu]")
    #plt.show()

    for i in range(0, len(arr)):
        arr[i] = arr[i] * hamming[i]
    
    return arr

def idft(array):
    fill = numpy.zeros(1024-len(array))
    array = numpy.append(array, fill)
    idft_arr = []
    arr_len = len(array)
    for i in range(arr_len):
        coef = 0
        for j in range(arr_len):
            coef += array[j]*cmath.exp(2j*cmath.pi*i*j*(1/arr_len))
        coef /= arr_len
        idft_arr.append(coef)
    return idft_arr

def dft(array):
    fill = numpy.zeros(1024-len(array))
    array = numpy.append(array, fill)
    dft_arr = []
    arr_len = len(array)
    for i in range(arr_len):
        coef = 0
        for j in range(arr_len):
            coef += array[j] * cmath.exp(-2j * cmath.pi * i * j * (1 / arr_len))
        dft_arr.append(coef)
    return dft_arr

def get_f0(serie, c0):
    c0_int = int(round(c0))
    series = numpy.copy(serie)
    if(numpy.argmax(serie[c0_int:]) != 0):
        return 16000 / numpy.argmax(serie[c0_int:])
    else:
        return 0

def centralclip(serie):
    series = numpy.copy(serie)
    max_val = numpy.abs(serie).max()
    
    n = len(series)

    for i in range(0, n):
        if(series[i] > max_val * 0.7):
            series[i] = 1
        elif(series[i] < -1 * max_val * 0.7):
            series[i] = -1
        else:
            series[i] = 0
    return series

def autocf(serie):
    series = numpy.copy(serie)
    n = len(series)
     
    data = numpy.asarray(series)

    mean = numpy.mean(data)
    c0 = numpy.sum((data - mean) ** 2)
    
    def r(h):
        acf_lag = ((data[:n - h]) * (data[h:])).sum()
        return round(acf_lag, 3)
    x = numpy.arange(n)
    acf_coeffs = list(map(r, x))

    c0_idx = int(round(c0))
    max_coef = numpy.argmax(acf_coeffs[c0_idx:])
    max_y = numpy.max(acf_coeffs[c0_idx:])
    return acf_coeffs, c0, max_coef+c0_idx, max_y

def maxSubArraySum(a,size): 
       
    max_so_far = -maxint - 1
    max_ending_here = 0
       
    for i in range(0, size):
        max_ending_here = max_ending_here + a[i] 
        if (max_so_far < max_ending_here): 
            max_so_far = max_ending_here 
  
        if max_ending_here < 0: 
            max_ending_here = 0   
    return max_so_far

Fs_toneoff, data_toneoff = read('../audio/maskoff_tone.wav')
Fs_toneon, data_toneon = read('../audio/maskon_tone.wav')

#plt.figure()
#plt.plot(data_toneoff)
#plt.xlabel('Sample index')
#plt.ylabel('Amplitude')
#plt.title('Waveform of test audio recording')
#plt.show()

#plt.figure()
#plt.plot(data_toneon)
#plt.xlabel('Sample index')
#plt.ylabel('Amplitude')
#plt.title('Waveform of test audio recording')
#plt.show()


# vystrihnutie sekundového úseku
t1 = 1.0 * 1000 #Works in milliseconds
t2 = 2.0 * 1000
newAudio = AudioSegment.from_wav("../audio/maskoff_tone.wav")
newAudio = newAudio[t1:t2]
newAudio.export('../audio/maskoff_tone_cut.wav', format="wav")

Fs_toneoff_cut, data_toneoff_cut = read('../audio/maskoff_tone_cut.wav')

# korelácia vystrihnutého úseku
correlation = numpy.correlate(data_toneon, data_toneoff_cut, "same")

max = 0
max_idx = 0
tmp = 0

cut_size = data_toneoff_cut.size
cor_size = correlation.size

for i in range(0, cut_size):
    max = max + correlation[i]

tmp = max

# získanie indexu začiatku najpodobnejšej sekundy
for j in range(cut_size + 1, cor_size):
    tmp = tmp + correlation[j] - correlation[j - cut_size]
    if (tmp > max):
        max = tmp
        max_idx = j

toneon_dur = librosa.get_duration(filename='../audio/maskon_tone.wav')

t1 = ((toneon_dur / data_toneon.size) * (max_idx - cut_size) * 1000) + 10
t2 = (t1 + 1000)
newAudio = AudioSegment.from_wav("../audio/maskon_tone.wav")
newAudio = newAudio[t1:t2]
newAudio.export('../audio/maskon_tone_cut.wav', format="wav")

Fs_toneon_cut, data_toneon_cut = read('../audio/maskon_tone_cut.wav')

data_toneoff_cut_w = numpy.zeros(data_toneoff_cut.size)
data_toneon_cut_w = numpy.zeros(data_toneon_cut.size)

#####################################################################
#            ****          ustrednenie    ****                      #
#####################################################################

for k in range(0, data_toneoff_cut.size):
    data_toneoff_cut_w[k] = data_toneoff_cut[k] - numpy.mean(data_toneoff_cut)
    data_toneon_cut_w[k] = data_toneon_cut[k] - numpy.mean(data_toneon_cut)

#####################################################################
#            ****          normalizacia    ****                     #
#####################################################################

for k in range(0, data_toneoff_cut.size):
    data_toneoff_cut_w[k] = data_toneoff_cut[k] / numpy.abs(data_toneoff_cut).max()
    data_toneon_cut_w[k] = data_toneon_cut[k] / numpy.abs(data_toneon_cut).max()

#plt.figure()
#plt.plot(data_toneoff_cut_w)
#plt.xlabel('čas')
#plt.ylabel('y')
#plt.title('Normalizácia a ustrednenie')
#plt.show()

#####################################################################
#           ****    rozsekanie na ramce - toneoff    ****           #
#####################################################################

length_toneoff = len(data_toneoff_cut_w)

window_hop_length_toneoff = 0.01 # dĺžka prekrytia rámcov v sekundách

overlap_toneoff = int(Fs_toneoff_cut * window_hop_length_toneoff)

window_size_toneoff = 0.02 # dĺžka rámca
framesize_toneoff = int(window_size_toneoff * Fs_toneoff_cut)

number_of_frames_toneoff = (length_toneoff // overlap_toneoff);

frames_toneoff = numpy.ndarray((number_of_frames_toneoff, framesize_toneoff)) # deklarácia 2D poľa pre jednotlivé rámce

# cyklus pre rozdelenie do rámcov
for k in range(0, number_of_frames_toneoff):
    for i in range(0, framesize_toneoff):
        if((k * overlap_toneoff + i) < length_toneoff):
            frames_toneoff[k][i] = data_toneoff_cut_w[k * overlap_toneoff + i]
        else:
            frames_toneoff[k][i] = 0

#fig_toneoff = plt.figure(figsize=(20, 5))
#ax_toneoff = fig_toneoff.add_subplot(111)
#plt.xlim([0, 320])
#plt.plot(frames_toneoff[90])
#locs_toneoff, labels_toneoff = plt.xticks()
#labels_toneoff = [float(item)*0.0625 for item in locs_toneoff]
#plt.xticks(locs_toneoff, labels_toneoff) 
#plt.xlabel('Time[ms]')
#plt.ylabel('Amplitude')
#plt.title('Waveform of a frame(toneoff)')
#plt.show()

#####################################################################
#           ****    rozsekanie na ramce - toneon    ****            #
#####################################################################

length_toneon = len(data_toneon_cut_w)
window_hop_length_toneon = 0.01     # dĺžka prekrytia rámcov v sekundách
overlap_toneon = int(Fs_toneon_cut * window_hop_length_toneon)

window_size_toneon = 0.02 # dĺžka rámca
framesize_toneon = int(window_size_toneon * Fs_toneon_cut)

number_of_frames_toneon = (length_toneon // overlap_toneon);

frames_toneon = numpy.ndarray((number_of_frames_toneon, framesize_toneon)) # deklarácia 2D poľa pre jednotlivé rámce

# cyklus pre rozdelenie do rámcov
for k in range(0,number_of_frames_toneon):
    for i in range(0,framesize_toneon):
        if((k * overlap_toneon + i) < length_toneon):
            frames_toneon[k][i] = data_toneon_cut_w[k * overlap_toneon + i]
        else:
            frames_toneon[k][i] = 0

#fig_toneon = plt.figure(figsize=(20, 5))
#ax_toneon = fig_toneon.add_subplot(111)
#plt.xlim([0, 320])
#plt.plot(frames_toneon[90])
#locs_toneon, labels_toneon = plt.xticks()
#labels_toneon = [float(item)*0.0625 for item in locs_toneon]
#plt.xticks(locs_toneon, labels_toneon) 
#plt.xlabel('Time[ms]')
#plt.ylabel('Amplitude')
#plt.title('Waveform of a frame(toneon)')
#plt.show()

#####################################################################
#           ****          center cliping    ****                    #
#####################################################################

ccframes_toneon = numpy.ndarray((number_of_frames_toneon, framesize_toneoff))
ccframes_toneoff = numpy.ndarray((number_of_frames_toneoff, framesize_toneoff))

for i in range(0, number_of_frames_toneon):
    ccframes_toneon[i] = centralclip(frames_toneon[i])
    ccframes_toneoff[i] = centralclip(frames_toneoff[i])

#####################################################################
#           ****          autocorrelation    ****                   #
#####################################################################

acframes_toneon = numpy.ndarray((number_of_frames_toneon, framesize_toneoff))
acframes_toneoff = numpy.ndarray((number_of_frames_toneoff, framesize_toneoff))
c0_toneon = numpy.ndarray(number_of_frames_toneon)
c0_toneoff = numpy.ndarray(number_of_frames_toneoff)

for i in range(0, number_of_frames_toneon):
    #funkcia autocf vracia: autokorelovaný signál, prah, súradnice lagu
    acframes_toneon[i], c0_toneon[i], max_x_toneoff, max_y_toneon = autocf(ccframes_toneon[i])
    acframes_toneoff[i], c0_toneoff[i], max_x_toneon, max_y_toneon = autocf(ccframes_toneoff[i])

#####################################################################
#              ****      základná frekvencia      ****              #
#####################################################################

f0_toneoff = numpy.ndarray(number_of_frames_toneoff)
f0_toneon = numpy.ndarray(number_of_frames_toneon)

for i in range(0, number_of_frames_toneon):
    f0_toneoff[i] = get_f0(acframes_toneoff[i], c0_toneoff[i])
    f0_toneon[i] = get_f0(acframes_toneon[i], c0_toneon[i])

f0_toneoff_mean = numpy.mean(f0_toneoff)
f0_toneon_mean = numpy.mean(f0_toneon)
scat_toneoff = numpy.ndarray(number_of_frames_toneoff)
scat_toneon = numpy.ndarray(number_of_frames_toneon)

for i in range(0, number_of_frames_toneon):
    scat_toneoff[i] = (f0_toneoff[i] - f0_toneoff_mean) ** 2
    scat_toneon[i] = (f0_toneon[i] - f0_toneon_mean) ** 2

f0_toneoff = (1/number_of_frames_toneoff) * numpy.sum(scat_toneoff)
f0_toneon = (1/number_of_frames_toneon) * numpy.sum(scat_toneon)

#fig, ax = plt.subplots(4,figsize=(20, 10))
#plt.subplots_adjust(hspace = .5)

#ax[0].plot(toneoff_frame)
#ax[0].set_title('Rámec')
#ax[0].set(xlabel='Čas', ylabel='y')

#ax[1].plot(centralclip_toneoff)
#ax[1].set_title('Centrálne klipovane s 70%')
#ax[1].set(xlabel='Čas', ylabel='y')

#ax[2].plot(autocor_toneoff)
#ax[2].set_title('Autokorelácia')
#ax[2].set(xlabel='Vzorky', ylabel='y')
#line = ax[2].axvline(c0_toneoff_f, color='black')
#stem = ax[2].stem(max_x, max_y, 'red')
#ax[2].legend([line, stem],["prah", "lag"],loc="upper right")

#ax[3].plot(f0_toneoff, label='bez rúška')
#ax[3].plot(f0_toneon, label='s rúškom')
#ax[3].set_title('Základná frekvencia')
#ax[3].set(xlabel='Rámce', ylabel='f0')
#ax[3].legend(loc="upper right")

#plt.show()

#####################################################################
#        ****       fourierova transformácia    ****                #
#####################################################################

ftframes_toneon = numpy.ndarray((number_of_frames_toneon, 1024), dtype=numpy.complex_)
ftframes_toneoff = numpy.ndarray((number_of_frames_toneoff, 1024), dtype=numpy.complex_)

ftcframes_toneon = numpy.ndarray((number_of_frames_toneon, 512))
ftcframes_toneoff = numpy.ndarray((number_of_frames_toneon, 512))

for i in range(0, number_of_frames_toneon):
    ftframes_toneon[i] = numpy.fft.fft(frames_toneon[i], 1024)
    ftframes_toneoff[i] = numpy.fft.fft(frames_toneoff[i], 1024)
    #ftframes_toneon[i] = dft(frames_toneon[i])
    #ftframes_toneoff[i] = dft(frames_toneoff[i])
    
    # použitie Hammingovho okna
    ftframes_toneon[i] = hamming_window(ftframes_toneon[i])
    ftframes_toneoff[i] = hamming_window(ftframes_toneoff[i])
    
    # výpočet výkonového spektra
    for j in range(0, 512):
        ftcframes_toneon[i][j] = 10.0 * math.log((abs(ftframes_toneon[i][j]) ** 2), 10)
        ftcframes_toneoff[i][j] = 10.0 * math.log((abs(ftframes_toneoff[i][j]) ** 2), 10)

#plt.figure(figsize=(20,5))
#plt.plot(ftcframes_toneon[5])
#plt.xlabel('Čas')
#plt.ylabel('Frekvencia')
#plt.title('Spektrum rámca s aplikáciou Hammingovho okna')
#plt.show()

#####################################################################
#               ****        spektogramy      ****                   #
#####################################################################

#plt.figure(figsize=(20,5))
#spect_toneoff = plt.imshow(ftcframes_toneoff.T, extent=[0, 1, 0, 8000], origin ='lower', aspect = 'auto') 
#plt.xlabel('Čas')
#plt.ylabel('Frekvencia')
#plt.title('Spektogram pre tón bez rúška')
#plt.colorbar(spect_toneoff)
#plt.show()

plt.figure(figsize=(20,5))
spect_toneon = plt.imshow(ftcframes_toneon.T, extent=[0, 1, 0, 8000], origin ='lower', aspect = 'auto') 
plt.xlabel('Čas')
plt.ylabel('Frekvencia')
plt.title('Spektogram pre tón s rúškom')
plt.colorbar(spect_toneon)
plt.show()

#####################################################################
#        ****       frekvencna charakteristika    ****              #
#####################################################################

frequency_response_div = numpy.zeros((number_of_frames_toneon, 512), dtype=numpy.complex_)

for i in range(0, number_of_frames_toneon):
    for j in range(0, 512):
        frequency_response_div[i][j] = abs(ftframes_toneon[i][j] // ftframes_toneoff[i][j])

frequency_response_avg = numpy.zeros(512)
frequency_response_spectrum = numpy.zeros(512)

for i in range(0, 512):
    tmp = complex(0,0)
    for j in range(0, number_of_frames_toneon):
        tmp += frequency_response_div[j][i]
    frequency_response_avg[i] = abs(tmp.real) / float(number_of_frames_toneon)
    frequency_response_spectrum[i] = 10.0 * math.log((abs(frequency_response_avg[i]) ** 2), 10)

plt.figure(figsize=(20, 5))
plt.plot(frequency_response_avg)
plt.title('Frekvenčná charakteristika rúška(výkonové spektrum)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#####################################################################
#          ****       filtracia, inverse dft      ****              #
#####################################################################

#reverse_lib = numpy.fft.ifft(frequency_response_avg, 1024)
reverse = idft(frequency_response_avg)

#plt.figure(figsize=(20, 5))
#plt.plot(reverse_lib)
#plt.title('fft')
#plt.show()

#plt.figure(figsize=(20, 5))
#plt.plot(reverse)
#plt.title('Impulzná odozva')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#####################################################################
#                ****       simulácia       ****                    #
#####################################################################

simulation_tone = scipy.signal.lfilter(reverse, [1], data_toneoff)
scipy.io.wavfile.write("../audio/sim_maskon_tone_window.wav", 16000, simulation_tone.real)

#plt.figure(figsize=(20, 5))
#plt.plot(data_toneon)
#plt.plot(data_toneoff)
#plt.plot(simulation_tone)
#plt.title('DATA TONEOFF')
#plt.show()

Fs_sentenceoff, data_sentenceoff = read('../audio/maskoff_sentence.wav')
Fs_sentenceon, data_sentenceon = read('../audio/maskon_sentence.wav')

simulation_sentence = scipy.signal.lfilter(reverse, [1.0], data_sentenceoff)
scipy.io.wavfile.write("../audio/sim_maskon_sentence_window.wav", 16000, simulation_sentence.real)

fig, ax = plt.subplots(3,figsize=(20, 10))
plt.subplots_adjust(hspace = .5)

ax[0].plot(data_sentenceoff)
ax[0].set_title('Graf nahrávky bez rúška')
ax[0].set(xlabel='Čas', ylabel='Frekvencia')

ax[1].plot(data_sentenceon)
ax[1].set_title('Graf nahrávky s rúškom')
ax[1].set(xlabel='Čas', ylabel='Frekvencia')

ax[2].plot(simulation_sentence)
ax[2].set_title('Graf simulácie rúška')
ax[2].set(xlabel='Čas', ylabel='Frekvencia')

plt.show()
