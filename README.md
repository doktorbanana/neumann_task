# Lautsprecher-Frequenzganglinearisierung mit inversem FIR-Filter

Ein Python-Tool zur Korrektur des Frequenzgangs von Lautsprechern durch inverses Filterdesign. Erzeugt einen FIR-Filter zur Kompensation von Frequenzgangverzerrungen.

## Übersicht
**Ziel**: Design eines inversen FIR-Filters zur Kompensation des Lautsprecherfrequenzgangs.  

**Key Features**:
- **FIR-Filterdesign** mit zwei Methoden: Least-Squares (firls) und Fenstermethode (firwin2)
- **Butterworth-Bandpass** zur Begrenzung auf den Nutzbereich
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
        'fs': 48000,  # Sampling frequency
        'fir_taps': 255,  # Filter order
        'regularization': 1e-6, # Vermeidet Teilen durch Null
        'design_method': 'firls', # Designmethode ('firls' oder 'firwin2')
        'bandpass_type': 'butterworth', # Typ des Bandpass ('butterworth' oder 'null' für keinen Bandpass)
        'lowcut': 125,    # Untere Bandpass-Grenze
        'highcut': 20000,  # Obere Bandpass-Grenze,
        'order': 4  # Ordnung des Bandpass
    }
```

2. Ausführung:

```bash
python main.py
```

3. Ergebnisse:

- **Plot**: Plot des Frequenzgangs vor/nach Equalization
- **Filterkoeffizienten**: ```exports/FIR_Linearization_Filter.wav```

## Ergebnis

- ±1 dB Toleranz im Bereich ~250 Hz – 18 kHz (bei 511 Taps)
- Limitationen:
  - Abweichungen <~250 Hz 
  - Notch bei ~11 kHz

## Technische Details

Design-Entscheidungen:
- FIR vs. IIR: FIR für lineare Phase, und geringere Komplexität der Implementierung 
- Butterworth-Bandpass zur Begrenzung auf den Nutzbereich: Maximale Flachheit im Durchlassbereich
- Least-Square-Error Methode: geringerer Fehler bei höherer Rechenlast

## Ausblick

- Warped FIR für verbesserte Tiefenlinearität [(Ramos et al., 2008)](https://www.sciencedirect.com/science/article/abs/pii/S1051200408000092)
- Psychoakustische Gewichtung der Fehlerfunktion
