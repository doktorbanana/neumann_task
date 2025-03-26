"""
Frequenzganglinearisierung eines Lautsprechers mittels inversem FIR-Filter

Ansatz:
Das Programm linearisiert den Frequenzgang eines Lautsprechers durch Design eines FIR-Filters,
der die inverse Übertragungsfunktion des Lautsprechers approximiert. Da keine Phaseninformation
des Lautsprechers vorliegt, wird ein linearphasiges Filterdesign gewählt, das die Phasencharakteristik
nicht verändert.

Bandpass-Integration:
Der natürliche Bandpass-Charakter des Lautsprechers wird durch Anwendung eines Butterworth-Bandpasses
(100Hz-20kHz) auf die lineare Zielkurve berücksichtigt. Dies verhindert kritische Verstärkungen in
Frequenzbereichen außerhalb der physikalischen Betriebsgrenzen des Lautsprechers, die zu
Signalverzerrungen oder Hardware-Schäden führen könnten.

Filterdesign-Methoden:
Es wurden zwei Methoden implementiert:
1. Hamming-Fensterung: Einfache Fenstermethode zur Approximation der idealen Impulsantwort
2. Least-Square-Error (LSE): Numerisch optimierte Methode mit minimalem mittleren Fehlerquadrat

Die LSE-Methode zeigt bei Tests bessere Ergebnisse (geringere Abweichungen <1dB im Hauptfrequenzband),
bei akzeptablem Rechenaufwand (Offline-Berechnung).

Ergebnisse:
- Bei 511 Filterkoeffizienten (@48kHz Sampling) wird die Zielkurve im Bereich 250Hz-11kHz mit
  ±1dB-Toleranz erreicht
- Restabweichungen:
  * Tiefen unter ~250Hz
  * Notch bei ~11kHz

Ausblick/Verbesserungspotenzial:
- Latenz-Reduktion: Kaskadierte warped-FIR Strukturen (Ramos et al., 2008) könnten die Linearität in den Tiefen
  bei reduzierter Filterlänge verbessern. Das könnte das Problem unter ~250Hz beheben.
  Aus Zeitgründen konnte dieser Ansatz nicht weiter verflogt werden.
- IIR-Ansätze: Potenzial für effizientere Implementierung (geringere Koeffizientenanzahl), jedoch
  komplexeres Design und nicht-linearer Phasenverlauf
- Erweiterungen:
  * Unit-Tests für kritische Komponenten
  * Verbesserung der Fehlerbehandlung im Code
  * Psychoakustische Gewichtung der Fehlerfunktion für LSE


Hinweis: Die Filterkoeffizienten werden als 32-bit Float WAV-Datei exportiert und können direkt
in DSP-Systemen verwendet werden. Der Plot zeigt den simulierten Frequenzgang vor/nach Equalization.
"""

from core.linearizer import Linearizer

if __name__ == "__main__":

    config = {
        'fs': 48000,  # Sampling frequency
        'fir_taps': 255,  # Filter order
        'regularization': 1e-2,
        'design_method': 'firls',
        'bandpass_type': 'butterworth',
        'lowcut': 125,    # Untere Bandpass-Grenze
        'highcut': 20000,  # Obere Bandpass-Grenze,
        'order': 4  # Ordnung des Bandpass
    }

    linearizer = Linearizer(config)
    linearizer.load_data('data/spectrum.json')
    linearizer.design_target_curve()

    inverse_response = linearizer.calculate_inverse_response()
    linearizer.design_filter(inverse_response)

    equalized_db = linearizer.simulate_response()
    linearizer.plot_results(equalized_db)
    linearizer.export_results('exports/FIR_Linearization_Filter.wav')
