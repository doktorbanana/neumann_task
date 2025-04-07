"""
Visualisierung - Darstellung der Frequenzgänge

Klassen:
- ResponsePlotter: Vergleich von Original vs. Smoothed vs. Inversed vs. Equalized

"""
import matplotlib.pyplot as plt


# ---------------------------- Visualization ----------------------------
class ResponsePlotter:
    """Visualisiert Frequenzgänge."""

    @staticmethod
    def plot(frequencies, original_db, smoothed_db, inverse_db, equalized_db):
        plt.figure(figsize=(12, 6))
        plt.semilogx(frequencies, original_db, label='Original Measurement')
        plt.semilogx(frequencies, smoothed_db, label='Smoothed Measurement')
        plt.semilogx(frequencies, inverse_db, label='Inverse (with Regularization)', linestyle='--')
        plt.semilogx(frequencies, equalized_db, label='Equalized')
        plt.xlim(20, 20000)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid(True, which='both')
        plt.title('Frequency Response Before/After Equalization')
        plt.show()
