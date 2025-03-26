"""
Visualisierung - Darstellung der Frequenzgänge

Klassen:
- ResponsePlotter: Vergleich von Original vs. Equalized

"""
import matplotlib.pyplot as plt


# ---------------------------- Visualization ----------------------------
class ResponsePlotter:
    """Visualisiert Frequenzgänge."""

    @staticmethod
    def plot(frequencies, original_db, target_db, equalized_db):
        plt.figure(figsize=(12, 6))
        plt.semilogx(frequencies, original_db, label='Original')
        plt.semilogx(frequencies, target_db, label='Target', linestyle='--')
        plt.semilogx(frequencies, equalized_db, label='Equalized')
        plt.xlim(20, 20000)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid(True, which='both')
        plt.title('Frequency Response Before/After Equalization')
        plt.show()
