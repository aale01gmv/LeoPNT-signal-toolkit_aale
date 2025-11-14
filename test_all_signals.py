import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_signal

# ---------------------------------------------------------
# Lista de señales a testear
# ---------------------------------------------------------
tests = [
    ("E5", 1, "pilot"),
    ("E5", 1, "data"),
    ("E5", 2, "pilot"),
    ("E5", 2, "data"),

    ("S1", 1, "pilot"),
    ("S1", 1, "data"),
    ("S1", 1, "acq"),

    ("S1", 2, "pilot"),
    ("S1", 2, "data"),
]

# Satélite por defecto
SVID = 4
duration_ms = 1     # 1 ms (puedes subirlo)
fs = 62.5e6         # el mismo sampling que usas

# ---------------------------------------------------------
# Función auxiliar para mostrar cada señal
# ---------------------------------------------------------
def show_signal(sig, title, nsamples=2000):
    plt.figure(figsize=(10,4))
    plt.plot(np.real(sig[:nsamples]), label="Real")
    plt.plot(np.imag(sig[:nsamples]), label="Imag")
    plt.title(title + f"  (primeras {nsamples} muestras)")
    plt.xlabel("muestra")
    plt.ylabel("amplitud")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Ejecutar tests
# ---------------------------------------------------------
for system, version, comp in tests:
    print(f"\n🔵 Probando {system} v{version} – {comp} ...")

    try:
        sig = generate_signal(
            system=system,
            version=version,
            component=comp,
            duration_ms=duration_ms,
            SVID=SVID,
            fs=fs
        )

        print("  ✓ OK. Muestras generadas:", len(sig))
        title = f"{system} v{version} – {comp}"
        show_signal(sig, title)

    except Exception as e:
        print("  ❌ ERROR:", e)
