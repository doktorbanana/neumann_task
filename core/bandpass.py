"""
Bandpass-Komponenten - Frequenzbereichsbegrenzung

Klassen:
- IBandpassFilter: Interface für alle Bandpass-Implementierungen
- NullBandpass: keine Frequenzbegrenzung000
- ButterworthBandpass: Butterworth-Filter mit konfigurierbarer Ordnung
- BandpassFactory: Erzeugt Bandpass-Instanzen nach Konfiguration

Funktionen:
- Begrenzung des Equalization-Bereichs auf physikalisch sinnvolle Frequenzen
- Verhindert Übersteuerung außerhalb des Nutzbereichs
- Butterworth-Charakteristik für maximale Flachheit im Durchlassbereich
"""

import numpy as np
from scipy.signal import butter, sosfreqz
from abc import ABC, abstractmethod

# ---------------------------- Bandpass ----------------------------


class IBandpassFilter(ABC):
    """Abstrakte Schnittstelle für Bandpass-Filter"""

    @abstractmethod
    def configure(self, **config):
        """Konfiguriert den Bandpass mit Parametern"""
        pass

    @abstractmethod
    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """Berechnet den Frequenzgang"""
        pass


class NullBandpass(IBandpassFilter):
    """Konkrete Implementierung für 'kein Bandpass'"""

    def configure(self, **config):
        pass

    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies)


class ButterworthBandpass(IBandpassFilter):
    """Konkrete Butterworth-Implementierung"""
    def __init__(self, fs):
        self.fs = fs
        self.sos = None

    def configure(self, lowcut, highcut, fs, order=4, **_):
        if lowcut >= highcut:
            raise ValueError("Highcut frequency must be above Lowcut frequency")

        self.sos = butter(
            N=order,
            Wn=[lowcut, highcut],
            btype='bandpass',
            fs=fs,
            output='sos')

    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        _, h = sosfreqz(self.sos, worN=frequencies, fs=self.fs)
        return np.abs(h)


class BandpassFactory:
    """Erzeugt den passenden Bandpass für gegebenen Typ"""
    _types = {
        'null': NullBandpass,
        'butterworth': ButterworthBandpass
    }

    @classmethod
    def get_bandpass(cls, config) -> IBandpassFilter:
        bandpass_type = config['bandpass_type'].lower()
        bandpass_class = cls._types.get(bandpass_type)

        if not bandpass_class:
            raise ValueError(f"Unsupported type: {bandpass_type}. Available: {list(cls._types.keys())}")

        # Instanziieren mit benötigten Parametern
        bandpass = bandpass_class(config['fs'])

        # Konfiguration mit allen relevanten Parametern
        bandpass.configure(**config)

        return bandpass
