import wave, struct
import numpy as np

fs = 44100
f = 220.0
duration = 10.0
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

sinus = lambda f, t: np.sin(2 * np.pi * f * t)
impulse = lambda f, t: np.where((f*t)%1 < 0.5, 1.0, -1.0)
triangle = lambda f, t: (2/np.pi) * (np.abs(((2*np.pi*f*t + 1.5*np.pi) % (2*np.pi)) - np.pi) - (np.pi/2))
saw = lambda f, t: (2*np.pi*f*t % (2*np.pi)) / np.pi - 1
noise = lambda f, t: np.random.uniform(-1, 1, len(t))

def write_wavefile(filename, signal, fs):
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1) # mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(fs) # samples per second
        data = (signal * 32767).astype(np.int16) # convert to 16-bit PCM
        wf.writeframes(data.tobytes())

# 1a. Сгенерировать звуковые сигналы различной формы:
# − синусоида;
write_wavefile('out_wav/sine.wav', sinus(f, t), fs)

# − импульс с различной скважностью;
write_wavefile('out_wav/impulse.wav', impulse(f, t), fs)

# − треугольная;
write_wavefile('out_wav/triangle.wav', triangle(f, t), fs)

# − пилообразная;
write_wavefile('out_wav/saw.wav', saw(f, t), fs)

# − шум.
write_wavefile('out_wav/noise.wav', noise(f, t), fs)

# 1б. Сгенерировать полифонические сигналы на основе сигналов из предыдущего пункта (суммировать несколько монофонических сигналов).
poly_signal = (sinus(440.0, t) + sinus(550.0, t) + sinus(660.0, t) + impulse(100.0, t)) 
poly_signal /= np.max(np.abs(poly_signal))
write_wavefile('out_wav/polyphonic.wav', poly_signal, fs)

# 1в. Сгенерировать звуковые сигналы с модуляцией параметров (ампли-
# туда, частота) несущих сигналов, полученных в 1а при помощи модулирую-
# щих сигналов различной формы:
base = sinus(f, t)

# − синусоида;
#AM
signal = (0.2 + 0.8 * sinus(2, t)) * base
write_wavefile('out_wav/am_modulated_sine.wav', signal, fs)

#FM
the_f = 1000 + 800 * (sinus(2, t) + 1)/2
phase = 2 * np.pi * np.cumsum(the_f) / fs
fm_sinus = np.sin(phase)
write_wavefile('out_wav/fm_modulated_sine.wav', fm_sinus, fs)

# − импульс с различной скважностью;
#AM
signal = (0.2 + 0.8 * impulse(2, t)) * base
write_wavefile('out_wav/am_modulated_impulse.wav', signal, fs)

#FM
the_f = f + 100 * impulse(2, t)
phase = 2 * np.pi * np.cumsum(the_f) / fs
fm_impulse = np.sign(np.sin(phase))
write_wavefile('out_wav/fm_modulated_impulse.wav', fm_impulse, fs)

# − треугольная;
#AM
signal = (0.2 + 0.8 * triangle(2, t)) * base
write_wavefile('out_wav/am_modulated_triangle.wav', signal, fs)

#FM
the_f = f + 100 * triangle(2, t)
phase = np.cumsum(the_f) / fs
fm_triangle = 2*np.abs(2*phase%1 - 1) - 1
write_wavefile('out_wav/fm_modulated_triangle.wav', fm_triangle, fs)

# − пилообразная.
#AM
signal = (0.2 + 0.8 * saw(2, t)) * base
write_wavefile('out_wav/am_modulated_saw.wav', signal, fs)

#FM
the_f = f + 100 * saw(2, t)
phase = np.cumsum(the_f) / fs
fm_saw = 2*phase%1 - 1
write_wavefile('out_wav/fm_modulated_saw.wav', fm_saw, fs)
