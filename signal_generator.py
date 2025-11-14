# signal_generator.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import scipy.signal as signal
from leo_spreading_codes import gen_primary_code

"""
Signal generator module.
Use:
    from signal_generator import generate_signal
    s = generate_signal(system="E5", version=1, component="pilot", duration_ms=1, SVID=1)
Returns:
    s_ref_demod : numpy array (complex)
"""
import numpy as np

# ---------------------------
# CS100 codes
# ---------------------------
CS100_CODES = {
    1:  "AB4146F921980A52E99FA781A",
    2:  "CFD1829A5AE6487E18A011912",
    3:  "A52664BD7FE03B9C7AAC26C34",
    4:  "CCD6AFF06ABEBD7234604D61A",
    5:  "DD14CEB418497EA7DEB0AF1F4",
    6:  "909E76F3DAA4F1EAFFC4355C9",
    7:  "AE43AD6DDEE764D87A73BE082",
    8:  "A0968DDC9DD27ADF370E9477B",
    9:  "C76E1B9EA0165A3C53F746E5C",
    10: "C782143D82315D99FB4B34813",
    11: "ACF12BE183AA684645FEF7B71",
    12: "AC3B8C74B31FB47C8ABE08095",
    13: "BB5BD982BBAC4D31E3D5E6F14",
    14: "F0843CAC935E5BB9B1DF85FB5",
    15: "D54AD85E324FEF7EE1AE9A8E4",
    16: "DF12AFA7D9FD5624F0D505B39",
    17: "A23848E174BBBF41939957535",
    18: "A7DFBC51B12FF08A7A581988A",
    19: "FB375539F4CAD6F19CE124D0F",
    20: "FD31AA32BB4CB8A843FB1CB65",
    21: "CEC68C5FA0F6D5FCF2057345D",
    22: "DFDC07AA6CF3F658CCA9C1A86",
    23: "8D0B6FA1A0DE29D2337497BB3",
    24: "CF8704D0AD5B68B999A738AFE",
    25: "CD351AC8D01CDB140E05EE0C7",
    26: "BB8118550E5BDA392CE213449",
    27: "8A5FDA0C8290385B1623DD2F9",
    28: "D0F511AB8AE10CC8242FBCF4C",
    29: "8463287D59A111B7E259B7055",
    30: "E29521759E7E64FF8A503E696"
}

# ---------------------------
# Helper functions
# ---------------------------

def hex_to_chips(hex_string, max_len=None):
    """Convert hex string to chip sequence ±1."""
    # if empty string, return array([1])
    if hex_string is None or hex_string == "":
        return np.array([1], dtype=int)
    bin_string = ''.join(bin(int(c, 16))[2:].zfill(4) for c in hex_string)
    chips = np.array([1 if b == '1' else -1 for b in bin_string], dtype=int)
    if max_len is not None and len(chips) > max_len:
        chips = chips[:max_len]
    return chips

def spreading_code(t, Px, Rc, R_sc=0, phi_sc=0):
    """Spreading code c(t)."""
    if Rc <= 0:
        return np.ones_like(t)
    Tc = 1.0 / Rc
    chip_indices = np.floor(t / Tc).astype(int)
    c = Px[chip_indices % len(Px)]
    if R_sc and R_sc > 0:
        c = c * np.sign(np.sin(2 * np.pi * R_sc * t + phi_sc))
    return c

def data_sequence(t, Dx, Rd):
    """Data overlay D(t)."""
    if Rd <= 0:
        return np.ones_like(t)
    Td = 1.0 / Rd
    data_indices = np.floor(t / Td).astype(int) % len(Dx)
    return Dx[data_indices]

def frequency_offset(t, f0_X, Delta_f_X, Thop_X, Moffset_X=0, Nfbins_X=None, FHVEC_X=None, mode="linear"):
    """
    Compute instantaneous frequency f_X(t) for FH modes.
    """
    t = np.array(t, ndmin=1)
    if mode == "linear":
        if Nfbins_X is None:
            m_X = np.zeros_like(t)
        else:
            m_X = (np.floor((t + Moffset_X * Thop_X) / Thop_X) % Nfbins_X) - (Nfbins_X - 1) / 2
    elif mode == "vector":
        if FHVEC_X is None:
            raise ValueError("FHVEC_X must be defined for vector mode")
        indices = np.floor(t / Thop_X).astype(int) % len(FHVEC_X)
        m_X = np.array(FHVEC_X)[indices]
    else:
        raise ValueError("mode must be 'linear' or 'vector'")
    f_X_t = f0_X + m_X * Delta_f_X
    return f_X_t

def s_X(t, phi, f, Px, Rc, Dx, Rd, R_sc=0, phi_sc=0):
    """Complex baseband component s_X(t)."""
    c = spreading_code(t, Px, Rc, R_sc, phi_sc)
    D = data_sequence(t, Dx, Rd)
    # carrier term included if f nonzero; we will demodulate later
    return np.exp(1j * phi) * np.exp(1j * 2 * np.pi * f * t) * c * D

def FHVEC_rotado(n, FHVEC_x):
    """Rotate FHVEC by circshift(-n+1)."""
    return np.roll(FHVEC_x, shift=-(n - 1))

def subcarrier_freq(n, start, modulo, step_MHz):
    """Compute generic subcarrier frequency in MHz for index n."""
    n = np.array(n)
    f = (start + ((n - 1) % modulo)) * step_MHz
    return f

# ---------------------------
# generate_component (core)
# ---------------------------

def generate_component(t, hex_code, Rc, Rd, phi=0, f0=0, R_sc=0, phi_sc=0,
                       max_len=None, data_code=None, Delta_f=0, Thop_X=0, Moffset_X=0,
                       Nfbins_X=None, FHVEC_X=None, mode="linear"):
    """
    Generate a component signal and return (s, f_total, s_demod).
    s : complex RF-like signal (with exp(j2π f(t) t) term)
    f_total : instantaneous frequency array
    s_demod : s multiplied by exp(-j2π f_total t) -> baseband/demodulated
    """
    # compute instantaneous frequency (for FH modes)
    if Delta_f == 0 or Thop_X == 0:
        f_X_t = np.zeros_like(t) + f0
    else:
        f_X_t = frequency_offset(t, f0, Delta_f, Thop_X, Moffset_X, Nfbins_X, FHVEC_X, mode)
    Px = hex_to_chips(hex_code, max_len=max_len)
    if data_code is None:
        # create simple data vector of ones with length ~= t*Rd
        if Rd <= 0:
            Dx = np.ones(1, dtype=int)
        else:
            N = max(1, int(round(t.size * Rd)))
            Dx = np.ones(N, dtype=int)
    else:
        Dx = hex_to_chips(data_code)
    s = s_X(t, phi, f_X_t, Px, Rc, Dx, Rd)
    # demodulate by removing instantaneous carrier
    s_demod = s * np.exp(-1j * 2 * np.pi * f_X_t * t)
    return s, f_X_t, s_demod

# ---------------------------
# Predefined hex codes and configurations
# (I include the full hex strings that you provided)
# ---------------------------

# ---------------------------
# Top-level generate_signal function
# ---------------------------

def generate_signal(system="E5", version=1, component="pilot", duration_ms=1, SVID=4, fs=62.5e6):
    """
    Generate requested signal:
      system: "E5" or "S1"
      version: 1 or 2  (corresponds to E5#1 / E5#2; S1#1 / S1#2)
      component: "pilot" or "data" or "acq" (acquisition)
      duration_ms: duration in ms
      SVID: satellite id (for codes)
      fs: sampling frequency in Hz
    Returns:
      s_ref_demod : np.ndarray (complex) with requested duration
    """
    # time vector
    t_ref = np.arange(0, duration_ms/1000.0, 1.0/fs)

    # choose based on system/version/component
    system = system.upper()
    component = component.lower()

    if system == "E5":
        if version == 1:
            # E5#1 configs
            if component == "pilot":
                #hex_code = HEX_E5_PILOT_1  # main pilot hex
                hex_code=gen_primary_code(6, 14, 10230) # (Columna de prime_codes.csv, svid, longitud de P_x) 
                Rc = 10 * 1023 * 1000
                Rd = 1000
                data_code = CS100_CODES.get(SVID, None)
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=0, f0=0, max_len=10230, data_code=data_code)
                return s_demod
            elif component == "data":
                # choose one of the data hex you provided
                hex_code = gen_primary_code(2, 14, 1023)
                Rc = 1023 * 1000
                Rd = 500
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=np.pi/2, f0=0, max_len=1023)
                return s_demod
            elif component == "acq":
                # acquisition not clearly defined for E5#1 in your last block; return ones
                return np.ones_like(t_ref, dtype=complex)
            else:
                raise ValueError("component must be 'pilot' or 'data' for E5 version 1")

        elif version == 2:
            # E5#2 configs (frequency hopping vector example)
            # You provided an empty hex for E5#2 pilot/data in the block; we'll include mechanism
            # Use placeholders if empty
            # FHVEC example you provided earlier:
            FHVEC_x = [-24, +24, -12, +12, -13, +13, -1, +1, -23, +23, -11, +11, -14, +14, -2, +2,
                       -22, +22, -10, +10, -15, +15, -3, +3, -21, +21, -9, +9, -16, +16, -4, +4,
                       -20, +20, -8, +8, -17, +17, -5, +5, -19, +19, -7, +7, -18, +18, -6, +6]
            FHVEC_x = FHVEC_rotado(SVID, np.array(FHVEC_x))
            Nfbins_x = len(FHVEC_x) + 1
            Delta_f_X = 1.023e6
            Thop_X = 1e-3
            Moffset_X = 49
            mode = "vector"

            if component == "pilot":
                hex_code = gen_primary_code(3, 14, 1023)  # you provided empty; handle gracefully
                Rc = 1023 * 1000
                Rd = 1000
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=0, f0=0, max_len=1023,
                                                   Delta_f=Delta_f_X, Thop_X=Thop_X, Moffset_X=Moffset_X,
                                                   Nfbins_X=Nfbins_x, FHVEC_X=FHVEC_x, mode=mode)
                return s_demod
            elif component == "data":
                hex_code = gen_primary_code(2, 14, 1023)  # empty in your block
                Rc = 1023 * 1000
                Rd = 500
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=np.pi/2, f0=0, max_len=1023)
                return s_demod
            else:
                raise ValueError("component must be 'pilot' or 'data' for E5 version 2")
        else:
            raise ValueError("Unsupported E5 version; use 1 or 2")

    elif system == "S1":
        # S1 signals (you provided S1#1 data)
        if version == 1:
            if component == "pilot":
                hex_code = gen_primary_code(4, 14, 5115)
                Rc = 5 * 1023 * 1000
                Rd = 1000
                data_code = CS100_CODES.get(SVID, None)
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=0, f0=0, max_len=5115, data_code=data_code)
                return s_demod
            elif component == "data":
                hex_code = gen_primary_code(2, 14, 1023)
                Rc = 1023 * 1000
                Rd = 500
                # f0_d per your formula:
                f0_d = subcarrier_freq(SVID, start=-6, modulo=13, step_MHz=1.023) * 1e6
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=np.pi/2, f0=f0_d, max_len=1023)
                return s_demod
            elif component == "acq":
                hex_adq = gen_primary_code(1, 14, 341)
                Rc_adq = (1.0/3.0) * 1023 * 1000
                Rd_adq = 1000
                f0_adq = subcarrier_freq(SVID, start=-20, modulo=41, step_MHz=1.023/3) * 1e6
                _, _, s_demod = generate_component(t_ref, hex_adq, Rc_adq, Rd_adq, phi=np.pi/2, f0=f0_adq, max_len=341, data_code=ADQUISITION_CODE)
                return s_demod
            else:
                raise ValueError("component must be 'pilot', 'data' or 'acq' for S1 version 1")
        elif version == 2:
            # You included an S#2 example with vector hopping; implement similarly to E5#2
            # Provide example FHVEC rotation for S#2
            FHVEC_x = [-20, +20, -10, +10, -11, +11, -1, +1, -19, +19, -9, +9, -12, +12, -2, +2,
                       -18, +18, -8, +8, -13, +13, -3, +3, -17, +17, -7, +7, -14, +14, -4, +4,
                       -16, +16, -6, +6, -15, +15, -5, +5]
            FHVEC_x = FHVEC_rotado(SVID, np.array(FHVEC_x))
            Nfbins_x = len(FHVEC_x) + 1
            Delta_f_X = (1.0/3.0) * 1.023e6
            Thop_X = 1e-3
            Moffset_X = 43
            mode="vector"

            if component == "pilot":
                hex_code = gen_primary_code(1, 14, 341)
                Rc = (1.0/3.0) * 1023 * 1000
                Rd = 1000
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=0, f0=0, max_len=341,
                                                   Delta_f=Delta_f_X, Thop_X=Thop_X, Moffset_X=Moffset_X,
                                                   Nfbins_X=Nfbins_x, FHVEC_X=FHVEC_x, mode=mode)
                return s_demod
            elif component == "data":
                hex_code = gen_primary_code(2, 14, 1023)
                Rc = 1023 * 1000
                Rd = 500
                f0_d = subcarrier_freq(1, start=-6, modulo=13, step_MHz=1.023) * 1e6
                _, _, s_demod = generate_component(t_ref, hex_code, Rc, Rd, phi=np.pi/2, f0=f0_d, max_len=1023)
                return s_demod
            else:
                raise ValueError("component must be 'pilot' or 'data' for S1 version 2")
        else:
            raise ValueError("Unsupported S1 version; use 1 or 2")

    else:
        raise ValueError("Unsupported system. Use 'E5' or 'S1'.")

# ---------------------------
# End of module
# ---------------------------

if __name__ == "__main__":
    # quick local test
    s = generate_signal(system="E5", version=1, component="pilot", duration_ms=1, SVID=1)
    print("Generated E5 v1 pilot, samples:", s.shape)
    plt.figure()
    plt.plot(np.real(s[:1000]), label='Real part')  
    plt.plot(np.imag(s[:1000]), label='Imaginary part')
    plt.legend()
    plt.show()
