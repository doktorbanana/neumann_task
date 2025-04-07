# Lautsprecher-Frequenzganglinearisierung mit inversem FIR-Filter

Ein Python-Tool zur Korrektur des Frequenzgangs von Lautsprechern durch inverses Filterdesign. Erzeugt einen FIR-Filter zur Kompensation von Frequenzgangverzerrungen.

## Übersicht
**Ziel**: Design eines inversen FIR-Filters zur Kompensation des Lautsprecherfrequenzgangs.  

**Key Features**:
- **FIR-Filterdesign** mit zwei Methoden: Least-Squares (firls) und Fenstermethode (firwin2)
- **Butterworth-Bandpass** zur Begrenzung auf den Nutzbereich
- **Glättung der Messdaten** mit Fraktionalem Oktavband oder ERB
- **Regularisierung der Messdaten** mit Tikhonov
- **Auto-Notch-Detection** um Nullstellen in der Übertragungsfunktion des LS zu finden
- **Linearphasiges Design** keine Phasenverzerrungen / -korrektur
- **Automatischer Export** der Filterkoeffizienten als 32-bit Float WAV
- **Visualisierung** des Frequenzgangs vor/nach Equalization

---

## Installation
1. **Python 3.9.6+** erforderlich
2. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
3. **Beispieldaten**: data/spectrum.json enthält Messdaten des Lautsprechers

## Verwendung

1. Konfiguration (in src/main.py anpassbar):
```bash
        config = {
        # Kerneinstellungen
        'fs': 48000,  # Abtastfrequenz in Hz
        'fir_taps': 255,  # Anzahl der Filterkoeffizienten (Filterordnung)
        'design_method': 'firls',  # Design-Methode: 'firls' oder 'firwin2'

        # Bandpass-Konfiguration
        'bandpass_type': 'butterworth',  # Filtertyp: 'butterworth' oder 'null'
        'bandpass_params': {
            'lowcut_freq': 125,  # Untere Grenzfrequenz in Hz
            'highcut_freq': 20000,  # Obere Grenzfrequenz in Hz
            'lowcut_order': 4,  # Ordnung des Tiefpassfilters (24 dB/Okt)
            'highcut_order': 2  # Ordnung des Hochpassfilters (12 dB/Okt)
        },

        # Glättungsparameter
        'smoothing_type': 'erb',
        # Glättungsmethode: 'octave', 'erb' oder 'null'
        'smoothing_params': {  # Zusatzparameter für Glättung
        },

        # Inversionsparameter
        'inverse_method': 'tikhonov',  # 'tikhonov' oder 'compare_squeeze'
        'inverse_params': {
            'beta': 0.1,  # Tikhonov-Dämpfungsfaktor (0.01-1.0)
            'b_filter_type': 'highpass',
            # Filtertyp für B(ω): 'highpass' oder 'lowpass'
            'cutoff_hz': 8000,  # Grenzfrequenz des Regularisierungsfilters
            'order': 2,  # Ordnung des B(w)-Filters
            'fs': 48000  # Abtastrate für Filterdesign
        },

        # Notch-Masking Parameter
        'notch_masking_type': 'prominence',  # 'prominence' oder 'null'
        'notch_masking_params': {
            'attenuation_db': 10.0,  # Dämpfung in Notch-Bereichen
            'min_depth_db': 6.0,  # Mindest-Notchtiefe zur Detektion
            'prominence': 3.0,  # Relative Prominenz zur Peakerkennung
            'rel_height': 0.5,  # Relative Höhe für Breitenberechnung
            'smooth_fraction': 3
            # Glättungsstärke (1=1/1 Oktave, 3=1/3 Oktave)
        }
    }
    }
```

2. Ausführung:

```bash
python main.py
```

3. Ergebnisse:

- **Plot**: Plot des Frequenzgangs vor/nach Equalization ```exports/255taps_fir_result.png```
- **Filterkoeffizienten**: 32bit float .wav ```exports/FIR_Linearization_Filter.wav```

## Ergebnis

- ±1 dB Toleranz im Bereich ab ~250 Hz
- Limitationen: Abweichungen <~250 Hz 

## Technische Details

Design-Entscheidungen:
- FIR vs. IIR: FIR für lineare Phase, und geringere Komplexität der Implementierung 
- Butterworth-Bandpass zur Begrenzung auf den Nutzbereich: Maximale Flachheit im Durchlassbereich
- Least-Square-Error Methode: geringerer Fehler bei höherer Rechenlast

## Ausblick

- Warped FIR für verbesserte Tiefenlinearität [(Ramos et al., 2008)](https://www.sciencedirect.com/science/article/abs/pii/S1051200408000092)
