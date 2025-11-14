#Author: FFEM

import numpy as np
import pandas as pd


def gen_start_value (oct_value, R): # PENDING! To handle error in case oct_value is not an octal (to check if the string starts with '0o')

    oct_value = bin(oct_value)

    mult3 = len(oct_value)+ (3-len(oct_value)%3)%3
    oct_value = oct_value[2:len(oct_value)].zfill(mult3) # Padding zeros are added until multiple of 3 (it's an octal value)
    if len(oct_value) < R:
        oct_value = oct_value.zfill(R)
    else:
        oct_value = oct_value[len(oct_value)-R:len(oct_value)] # Value is truncated to fit length to the number of registers
    oct_value = [int(i) for i in oct_value] 

    return oct_value

def gen_taps (oct_tap, R): # Polynomial coefficients for the LFSR. R is the number of registers in the LFSR. 

    oct_tap = bin(oct_tap)

    mult3 = len(oct_tap)+ (3-len(oct_tap)%3)%3
    oct_tap = oct_tap[2:len(oct_tap)].zfill(mult3) # 2 first elements are discarded because they only represent the format of the parameter, and we add padding 0 until multiple of 3 (it's an octal value)
    oct_tap = oct_tap[0:len(oct_tap)-1] # Last bit is discarded (as per specification)
    oct_tap = oct_tap[len(oct_tap)-R:] # Array is truncated to have the same number of elements than registers in the LFSR
    oct_tap = oct_tap[::-1] # String is reversed (in specification last bits have lowest index)
    index = np.array([i+1 for i in range(len(oct_tap)) if oct_tap[i] == '1'] [::-1]) # Indices in the array of elements equal to 1. Then the array is reversed (maybe it is not mandatory to do so)
   
    return index

def gen_prim_code_bin(R, NoChips, taps1_oct, taps2_oct, start_value1, start_value2): #Implementation of LFSR 

    taps1 = gen_taps(taps1_oct,R)
    taps2 = gen_taps(taps2_oct,R)
    c1 = gen_start_value(start_value1,R)
    c2 = gen_start_value(start_value2,R)
    primary_code = ''.join(map(str,(((np.logical_xor(c1,c2))[0:NoChips]).astype(int)))) #In the first cycle, the output is the XOR of the 2 initial sequences

    for j in range(0,NoChips):
        xor1 = 0 
        xor2 = 0
        for i in taps1:
            xor1 ^= (c1[j:R+j])[R-i] # R-i because index are reversed (in Python index 0 is at the left and in specification the left bit has the max index)
        for k in taps2:
            xor2 ^= (c2[j:R+j])[R-k]

        c1.append(xor1)
        c2.append(xor2)

        primary_code += (str)((int)(xor1^xor2))

    primary_code = primary_code[0:NoChips] 
    characs_left = (4-(NoChips%4))%4 # El número de bits debe ser múltiplo de 4 para la conversión a hexadecimal, así que añadimos 0 hasta completar un múltiplo de 4
    characs_total = NoChips + characs_left
    for i in range(0,characs_left):
        primary_code += '0'

    return primary_code

def bin_to_hex_code (NoChips, pc):

    hex_primary_code = ''

    characs_left = (4-(NoChips%4))%4 # Number of bits must be multiple of 4, so we do zero-padding
    characs_total = NoChips + characs_left

    for i in range(0,int(characs_total/4)):
        hex_primary_code += hex(int(pc[4*i:4*(i+1)],2))[2:] # [2:] to discard the first 2 elements that indicate the format of the parameter. 

    hex_primary_code = hex_primary_code.upper()
    
    return hex_primary_code

def overlayCode_bin (n):
    
    f = open("leo_overlay_codes.txt", "r")
    second_code = f.readlines()[n-1]
    second_code_bin = bin(int(second_code, 16))[2:]

    return second_code_bin

def gen_tiered_code(pcode_bin, scode_bin, length_s, chiprate): # length_s: length in seconds
    
    tieredCode = np.array([],dtype=bool)

    nchips = np.ceil(length_s*chiprate) # number of chips in length_s[seconds] 
    nIntegerPCode = (int)(np.floor(nchips/len(pcode_bin))) # Number of entire primary codes in length_s 
    nRestPCode = (int)(nchips - nIntegerPCode*len(pcode_bin)) # Rest of bits to compute 

    pcode_bin_aux = np.array(list(pcode_bin), dtype=int).astype(bool)

    for i in range(0,nIntegerPCode):
        second_aux = np.repeat((int)(scode_bin[i]),len(pcode_bin)).astype(bool)
        tieredCode = np.append(tieredCode, np.logical_xor(pcode_bin_aux,second_aux))

    second_aux = np.repeat((int)(scode_bin[nIntegerPCode]),nRestPCode).astype(bool)
    tieredCode = np.append(tieredCode,np.logical_xor(pcode_bin_aux[0:nRestPCode],second_aux))

    tieredCode = -1+2*tieredCode.astype(int)

    return tieredCode

def generate_code(R, NoChips, taps1_oct, taps2_oct, start_value1, start_value2): # FUNCTION TO BE CALLED FROM OUTSIDE
    return bin_to_hex_code(NoChips, gen_prim_code_bin(R, NoChips, taps1_oct, taps2_oct, start_value1, start_value2))

def tieredcode_to_string (code):
    code_str = ''.join(map(str,(code)))
    return code_str

def write_to_file(name, code): # code should be a string

    f = open(name,"w")
    f.write(code)
    f.close()

prim_code_info = pd.read_csv("prim_codes.csv", sep=';')
def gen_primary_code(family, svid, L):
    
    code_L = (int)(prim_code_info.iloc[0,family-1])
    feedback_1 = int(str((int)(prim_code_info.iloc[1,family-1])), 8) # int(a,base) needs "a" to be a string
    feedback_2 = int(str((int)(prim_code_info.iloc[2,family-1])), 8)
    start_value_1 = int(str((int)(prim_code_info.iloc[3,family-1])), 8)
    start_value_2 = int(str((int)(prim_code_info.iloc[3+svid,family-1])), 8)
    
    return generate_code(code_L, L, feedback_1, feedback_2, start_value_1, start_value_2)

##### Some testing 

L=1023
codigo_hex = gen_primary_code(3, 14, L) # (Columna de prime_codes.csv, svid, longitud de P_x) 

print(codigo_hex)





