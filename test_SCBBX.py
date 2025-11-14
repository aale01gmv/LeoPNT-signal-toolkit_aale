import runpy
import numpy as np

# Ejecuta el script y obtiene su espacio de variables
script_path = r"C:\Users\aale\OneDrive - gmv.com\Desktop\Python pruebas\Señales_CL0\SCB_aale.py"  # Cambia la ruta a tu script
result = runpy.run_path(script_path)

# Accedemos a SCBBX
SCBBX = result.get("SCBBX", None)

if SCBBX is None:
    print("[❌ ERROR] No se encontró la variable SCBBX en el script.")
else:
    max_SCBBX = np.max(SCBBX)
    print(f"SCBBX máximo = {max_SCBBX:.1f} ps")
    
    if max_SCBBX <= 350:
        print("[✅ OK] SCBBX máximo dentro del límite de 350 ps")
    else:
        print("[❌ ALERT] SCBBX máximo supera 350 ps")
    
    assert max_SCBBX <= 350, f"SCBBX máximo ({max_SCBBX:.1f} ps) supera 350 ps"
