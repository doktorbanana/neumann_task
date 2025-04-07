"""
Frequenzganglinearisierung eines Lautsprechers mittels inversem FIR-Filter

Ansatz:
Das Programm linearisiert den Frequenzgang eines Lautsprechers durch Design eines FIR-Filters,
der die inverse Übertragungsfunktion des Lautsprechers approximiert. Da keine Phaseninformation
des Lautsprechers vorliegt, wird ein linearphasiges Filterdesign gewählt, das die Phasencharakteristik
nicht verändert.

Bandpass-Integration:
Der natürliche Bandpass-Charakter des Lautsprechers wird durch Anwendung eines Butterworth-Bandpasses
(125Hz-20kHz) auf die lineare Zielkurve berücksichtigt. Dies verhindert kritische Verstärkungen in
Frequenzbereichen außerhalb der physikalischen Betriebsgrenzen des Lautsprechers, die zu
Signalverzerrungen oder Hardware-Schäden führen könnten.
Eine typische Roll-Off-Rate für Lautsprecher mit Bassreflexöffnung ist 24dB/Oktave für tiefe Frequenzen.
Das ist auch in den vorliegenden Messdaten erkennbar. Daher wurde für den Lowcut ein Filter vierter Ordnung gewählt.
Im hochfrequenten Bereich ist der Roll-Off in der Regel flacher. Daher wurde ein Highcut-Filter
zweiter Ordnung gewählt.

Automatische Notch-Detektion und Auffüllen der Notches:
Um Nullstellen in der Übertragungsfunktion des Lautsprechers zu finden, wurde eine automatische Notch-Detection
implementiert. In den Beispieldaten wird ein Notch bei 11kHz erkannt und für das Design des Filters ignoriert.
Die Detection basiert auf einem Vergleich mit einer 1/3-Oktavband geglätteten Version der Übertragungsfunktion.
An Stellen, an denen die geglättete Kurve deutlich unter dem Original liegt, liegt ein schmalbandiger Notch vor.
Dieser Notch wird dann 'aufgefüllt', d.h. die es wird verstärkt (10dB).

Regularisierung:
Mit der Tikhonov-Methode wird der Dämpfungsterm β·B(ω)² hinzugefügt, wobei B(ω) einem Hochpassfilter
(8 kHz Grenzfrequenz) entspricht. Verhindert Überverstärkung im hochfrequenten Bereich.


Filterdesign-Methoden:
Es wurden zwei Methoden implementiert:
1. Hamming-Fensterung: Einfache Fenstermethode zur Approximation der idealen Impulsantwort
2. Least-Square-Error (LSE): Numerisch optimierte Methode mit minimalem mittleren Fehlerquadrat

Die LSE-Methode zeigt bei Tests bessere Ergebnisse (geringere Abweichungen <1dB im Hauptfrequenzband),
bei akzeptablem Rechenaufwand (Offline-Berechnung).

Ergebnisse:
- Bei 255 Filterkoeffizienten (@48kHz Sampling) wird die Zielkurve im Bereich 250Hz-18kHz mit
  ±1dB-Toleranz erreicht
- Restabweichungen:
  * Tiefen unter ~250Hz

Ausblick/Verbesserungspotenzial:
- Latenz-Reduktion: Kaskadierte warped-FIR Strukturen (Ramos et al., 2008) könnten die Linearität in den Tiefen
  bei reduzierter Filterlänge verbessern. Das könnte das Problem unter ~250Hz beheben.
  Aus Zeitgründen konnte dieser Ansatz nicht weiter verflogt werden.
- IIR-Ansätze: Potenzial für effizientere Implementierung (geringere Koeffizientenanzahl), jedoch
  komplexeres Design und nicht-linearer Phasenverlauf.
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
        'fs': 48000,                     # Sampling frequency
        'fir_taps': 255,                 # Filter Ordnung
        'design_method': 'firls',        # Designmethode ('firls' oder 'firwin2')
        'bandpass_type': 'butterworth',  # Typ des Bandpass ('butterworth' oder 'null' für keinen Bandpass)
        'bandpass_params':
            {
                'lowcut_freq': 125,     # Untere Bandpass-Grenze
                'highcut_freq': 20000,  # Obere Bandpass-Grenze,
                'lowcut_order': 4,      # Ordnung des Lowcuts (untere Flanke des Bandbass)
                'highcut_order': 2,     # Ordnung des Highcuts (obere Flanke des Bandbass)
            },
        'smoothing_type': 'erb',  # Art der  Glättung ('octave', 'erb', oder 'null' für keine Glättung)
        'smoothing_params':
            {
            },
        'inverse_method': 'tikhonov',  # Methode zur Berechnung des inversen Amplitudengangs ('simple' oder 'tikhonov')
        'inverse_params':
            {
                'beta': 0.1,                    # Dämpfungsfaktor
                'b_filter_type': 'highpass',    # frequenzabhängige Dämpfungsfunktion ('highpass' oder 'lowpass')
                'cutoff_hz': 8000,              # Grenzfrequenz
                'order': 2,                     # Filterordnung
                'fs': 48000                     # Abtastrate
            },
        'notch_masking_type': 'prominence',     # Methode für Notch-Detection und -Masking ('null' oder 'prominence')
        'notch_masking_params':
            {
                'attenuation_db': 10.0,         # Wie stark sollen die Notches aufgefüllt werden
                'min_depth_db': 6.0,            # Mindesttiefe der detektierten Notches
                'prominence': 3.0,              # Prominenz der detektierten Notches
                'rel_height': 0.5,              #
                'smooth_fraction': 3
            }
    }

    linearizer = Linearizer(config)
    linearizer.load_data('data/spectrum.json')

    linearizer.mask_notches()
    linearizer.smooth_data()
    linearizer.design_target_curve()
    inverse_response = linearizer.calculate_inverse_response()
    linearizer.design_filter(inverse_response)

    equalized_db = linearizer.simulate_response()
    linearizer.plot_results(equalized_db)
    linearizer.export_results('exports/FIR_Linearization_Filter.wav')
