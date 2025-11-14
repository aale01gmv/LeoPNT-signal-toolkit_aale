import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import scipy.signal as signal

from signal_generator import generate_signal

# ========================
# Funciones auxiliares
# ========================

# --- Filtrado ideal ---
def brick_wall_filter(signal, fs, bw, f_central=0):
    """
    Filtro ideal tipo 'brick wall' centrado en f_central.

    Parámetros:
    ------------
    signal : array_like
        Señal de entrada (dominio temporal)
    fs : float
        Frecuencia de muestreo [Hz]
    bw : float
        Ancho de banda total del filtro [Hz]
    f_central : float
        Frecuencia central del filtro [Hz]
    
    Retorna:
    --------
    signal_filtered : array_like
        Señal filtrada en el dominio temporal
    S_freq : array_like
        Espectro original (FFT)
    freqs : array_like
        Vector de frecuencias (Hz)
    """
    N = int(fs*1e-3)
    # FFT y eje de frecuencias
    S_freq = np.fft.fftshift(np.fft.fft(signal,N))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    # Máscara del filtro centrado en f_central
    mask = np.logical_and(freqs > (f_central - bw/2), freqs < (f_central + bw/2))

    # Aplicar filtro
    S_filtered = S_freq * mask

    # Señal filtrada (dominio temporal)
    signal_filtered = np.fft.ifft(np.fft.ifftshift(S_filtered))
    
    return signal_filtered, S_freq, freqs









# ========================
# Parametros generales
# ========================

# Tiempo de simulación
t_total = 100/62.5
t_total_ref = 1e-3  # 1 ms
fs = 62.5*1e6      
t_time = np.arange(0, t_total, 1/fs)
t_ref = np.arange(0, t_total_ref, 1/fs)

SVID = 4  # Número de satélite (1-36)

# ========================
# Señal GRABADA 
# Ruta del archivo IQ
#filename = r"\\staascorp\legion\totem_B_recordings\Jan2025\signal_S1_92.16M_8bit_10min.iq"
#filename=r"\\staascorp\legion\PayloadAIV_TestResults\2025-10-29_15-30 - EMC DummyMsg OCXO-N ODTS-N config1 SVID14\01 - Results\16_43_20(Auto_C1_Dummy_N)\GSE\20251029AutoC1DummyN3T180s\ch1.dat"
filename = r"C:\Users\aale\OneDrive - gmv.com\Desktop\GSE\Pruebas ganancias 0511\LowGain2020Date05112025Config1\ch1.dat"

num_samples = 100e6

type = 'uint8'
signal_file = np.zeros(int(2 * num_samples), dtype=type)
signal_file_aux = np.fromfile(filename, dtype=type, count=int(2 * num_samples), offset=0)
signal_file[:] = signal_file_aux.astype('int8')
    
signal_file_aux_alt = np.zeros(int(len(signal_file) / 2), dtype=np.dtype([('re', np.int8), ('im', np.int8)]))
signal_file_aux_alt[:]['re'] = signal_file[::2]
signal_file_aux_alt[:]['im'] = signal_file[1::2]
iqs = signal_file_aux_alt.view(np.int8).astype(np.float32).view(np.complex64)

# Crear la señal compleja
S = iqs
"""
plt.figure(figsize=(5,5))
plt.scatter(np.real(S[::100]), np.imag(S[::100]), s=1, alpha=0.5)
plt.title("Constelación IQ")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axis('equal')
plt.grid(True)
plt.show()
"""

### ========================
# Generar señal E5#1
### ========================

###################################################################################3 A cambiar según señal a generar
############ Configuración 1 ############
bw_rx_p=20.46e6
bw_rx_d=2.046e6

sig_piloto = generate_signal(
            system="E5",
            version=1,
            component="pilot",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
sig_data =  generate_signal(
            system="E5",
            version=1,
            component="data",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )


"""
############ Configuración 2 ############

bw_rx_p = 51.15e6
bw_rx_d=2.046e6
sig_piloto = generate_signal(
            system="E5",
            version=2,
            component="pilot",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
sig_data =  generate_signal(
            system="E5",
            version=2,
            component="data",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
"""
"""

##################################################################################################################
### ========================
# Generar señal S1#1
### ========================


################# SVID 4, PRN 14  Config 1 ####################
bw_rx_p = 10.23e6
bw_rx_d=2.046e6
bw_rx_acq = 1.023e6

sig_piloto = generate_signal(
            system="S1",
            version=1,
            component="pilot",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
sig_data =  generate_signal(
            system="S1",
            version=1,
            component="data",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
sig_acq =  generate_signal(
            system="S1",
            version=1,
            component="acq",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
"""
"""
################# SVID 4, PRN 14 Config 2 ####################

bw_rx_p = 15e6
bw_rx_d=2.046e6
sig_piloto = generate_signal(
            system="S1",
            version=2,
            component="pilot",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )
sig_data =  generate_signal(
            system="S1",
            version=2,
            component="data",
            duration_ms=1,
            SVID=SVID,
            fs=fs
        )

"""

s_ref = sig_piloto
f_central=0
Rc=10.23e6  # chiprate E5#1 and S1#1
bw_rx = bw_rx_p

########################################################################################################

# --- Filtrado en banda base ---
filtered, S_freq, freqs = brick_wall_filter(S, fs, bw_rx, f_central)
S_demod_total=filtered*np.exp(-1j * 2 * np.pi * (f_central) * t_ref) # down-conversion to baseband


##################################################################################################
# --- Cross-Correlation Normalizada ---
def cross_correlation_normalized(s_PL_BX, s_REF, fs):
    dt = 1/fs
    corr = signal.correlate(s_PL_BX, s_REF, mode="full") #
    norm_factor = np.sqrt(np.sum(np.abs(s_PL_BX)**2) * np.sum(np.abs(s_REF)**2))
    corr_norm = corr / norm_factor
    lags = signal.correlation_lags(len(s_PL_BX), len(s_REF), mode="full") * dt
    return lags, corr_norm


lags,CCF = cross_correlation_normalized(S_demod_total,s_ref, fs)
plt.figure(figsize=(10,4))
plt.plot(lags, np.abs(CCF), label='Parte Real') 
plt.xlabel('Retardo [s]')   
plt.title("Correlación cruzada normalizada - Señal completa")
plt.show()

idx_max = np.argmax(np.abs(CCF))
lag_max = lags[idx_max]
muestras=int(lag_max*fs)
print(f"Retardo máximo: {muestras:.3f} muestras")
##################################################################################################


shift =muestras  #8125000+65485+1 #35587#35540  # ejemplo: desplazar 20000 muestras hacia la derecha
s_ref_rolled = np.roll(s_ref, shift)

# --- Recorte de la señal ---

t_inicio = 0e-3      # 0 ms
duracion = 1e-3      # 1 ms

# Convertir tiempo a número de muestras
muestra_inicio = int(t_inicio * fs)
num_muestras = int(duracion * fs)

# Recortar la señal
S_recortada = S_demod_total[muestra_inicio : muestra_inicio + num_muestras]


lags,CCF = cross_correlation_normalized(S_recortada,s_ref_rolled, fs)
plt.figure(figsize=(10,4))
plt.plot(lags, np.abs(CCF), label='Parte Real') 
plt.xlabel('Retardo [s]')   
plt.title("Correlación cruzada - Señal recortada")
plt.show()    

idx_max = np.argmax(np.abs(CCF))
print(f"Índice del máximo: {idx_max}")
lag_max = lags[idx_max]
print(f"Retardo máximo: {lag_max*1e3:.3f} ms")
muestras=int(lag_max*fs)
print(f"Retardo máximo: {muestras:.3f} muestras")

##-------Interpolación---------##
# Paso de interpolación deseado
dt_target = 5e-12  # picosegundos

# Crear función de interpolación (cúbica para suavidad)
interp_func = interpolate.interp1d(lags, np.abs(CCF), kind='cubic')

# Generar vector interpolado refinado:
# para cada intervalo entre puntos originales, añadimos submuestras
lags_interp = []
for i in range(len(lags)-1):
    # puntos entre lag[i] y lag[i+1]
    n_steps = int(np.ceil((lags[i+1]-lags[i]) / dt_target))
    if n_steps > 1:
        seg = np.linspace(lags[i], lags[i+1], n_steps, endpoint=False)
        lags_interp.append(seg)
lags_interp = np.concatenate(lags_interp + [lags[-1:]])
fs_interp=1/ (lags_interp[2]-lags_interp[1])
# Calcular correlación interpolada
CCF_interp = interp_func(lags_interp)

"""
##----------------------------------------##
# -----------------------------
#  PLOT COMPARATIVO
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(lags_recortada * 1e9, np.abs(CCF_recortada), 'o', label='Puntos originales', alpha=0.7)
plt.plot(lags_interp * 1e9, CCF_interp, 'o', label='Interpolado (~50 ps)', markersize=2, alpha=0.7)
plt.xlabel('Retardo [ns]')
plt.ylabel('|CCF|')
plt.title('Interpolación de la correlación conservando los puntos originales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# --- S-Curve ---
def S_curve(shift, CCF):
    """Calcula S-Curve para una separación early-late δ."""
    ccf1=np.roll(CCF,-shift)
    ccf2=np.roll(CCF,shift)
    return np.abs(ccf2)**2 - np.abs(ccf1)**2, ccf1, ccf2

def rango_acumulativo(eps_bias_vals_time):
    eps_bias_vals_time = np.array(eps_bias_vals_time)
    # Inicializamos arrays para el máximo y el mínimo acumulativo
    max_acum = np.maximum.accumulate(eps_bias_vals_time)
    min_acum = np.minimum.accumulate(eps_bias_vals_time)
    # Rango acumulativo: diferencia entre máximo y mínimo hasta cada punto
    SCBBX = max_acum - min_acum
    return SCBBX

# --- Cálculo del sesgo ε_bias,BX(δ) ---
def epsilon_bias(shift, CCF):
    S1, ccf1, ccf2 = S_curve(shift, CCF)  
# --- Vector de eje x (muestras relativas al centro) ---
# x = (np.arange(len(CCF)) - len(CCF)//2)*dt
    # =lags
    x1 =lags_interp

    #plt.figure(figsize=(10,4))      
    #plt.plot(x1, S1, label='Interpolado (~50 ps)')

    xi_max=np.argmax(S1)
    xi_min=np.argmin(S1)
    # --- TRUNCAR al rango deseado ---
    #mask = (x1 >= -0.5e-6) & (x1 <= 0.5e-6)
    #mask = (x1 >= xi_min) & (x1 <= xi_max)
    x = x1[xi_min:xi_max]
    S = S1[xi_min:xi_max]
    #ccf1 = ccf1[mask]
    #ccf2 = ccf2[mask]
    #plt.figure(figsize=(10,4))      
    #plt.plot(x, S, label='Interpolado (~50 ps)')
    
# Buscamos los puntos donde S cambia de signo
    sign_change = np.where(np.diff(np.sign(S)))[0]
    return np.abs(x[sign_change])

zero_nominal_all = []
max_zero = []
min_zero = []

# Variables acumuladas para máximo y mínimo de zero crossings
max_accum = 0
min_accum = np.inf
    #return np.abs(zero_crossings)
    

# --- Evaluar ε_bias para toda la gama δ ∈ (0, δmax] ---
delta_max=0.5
deltas = np.arange(0.01, delta_max, 0.01)
#deltas=[0.5]
print(f"Evaluando ε_bias para delta = {(deltas)} ")


for d in deltas:
    zero_crossings = epsilon_bias(d * fs_interp / Rc, CCF_interp)
    
    zero_nominal_all.append(zero_crossings*1e12)
    # Acumular máximo y mínimo de los zero crossings hasta delta actual
    if zero_crossings > max_accum: max_accum = zero_crossings
    if min_accum == np.inf: min_accum = zero_crossings #First loop
    else:
        if zero_crossings < min_accum: min_accum = zero_crossings
    max_zero.append(max_accum)
    min_zero.append(min_accum)
# Nominal zero crossings (take the first crossing if multiple exist), converted to ps
# nominal_zero_ps = np.array([z[0] if len(z) > 0 else np.nan for z in zero_nominal_all]) 

# --- Plot ---
plt.figure(figsize=(10,5))

# Accumulated maximum and minimum lines
plt.plot(deltas, np.array(max_zero)*1e12, label='Accumulated max zero crossing', color='blue')
plt.plot(deltas, np.array(min_zero)*1e12, label='Accumulated min zero crossing', color='orange')

# Nominal zero crossings (take the first crossing if multiple exist)
plt.scatter(deltas, zero_nominal_all,
            label='Nominal zero crossing', color='red', s=10)

plt.xlabel('Delta')
plt.ylabel('Zero crossing [ps]')
plt.title('Nominal zero crossings and accumulated max/min')
plt.grid(True)
plt.legend()
plt.show()

print(zero_nominal_all)
# --- Calcular el S-Curve Bias ---
SCBBX = rango_acumulativo(zero_nominal_all)


plt.figure(figsize=(8,4))
plt.plot(deltas, SCBBX, 'o-')
plt.title(f'S-Curve Bias SCBBX (ps)')
plt.xlabel(r'Early–Late spacing $\delta$ [chips]')
plt.ylabel(r'S-Curve Bias [ps]')
plt.grid(True)
plt.legend()
plt.show()
























































































