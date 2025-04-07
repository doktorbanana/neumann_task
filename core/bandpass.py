"""
Bandpass-Komponenten - Frequenzbereichsbegrenzung

Klassen:
- IBandpassFilter: Interface für alle Bandpass-Implementierungen
- NullBandpass: keine Frequenzbegrenzung
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
    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        """Berechnet den Frequenzgang"""
        pass


class NullBandpass(IBandpassFilter):
    """Konkrete Implementierung für 'kein Bandpass'"""

    def __init__(self):
        pass

    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies)


class ButterworthBandpass(IBandpassFilter):
    """Konkrete Butterworth-Implementierung"""
    def __init__(self,
                 fs: int,
                 lowcut_freq: int,
                 highcut_freq: int,
                 lowcut_order: int,
                 highcut_order: int
                 ):
        self.fs = fs

        if lowcut_freq >= highcut_freq:
            err_msg = "Highcut frequency must be above Lowcut frequency"
            raise ValueError(err_msg)

        self.highcut_sos = butter(
            N=highcut_order,
            Wn=highcut_freq,
            btype='lowpass',
            fs=self.fs,
            output='sos')

        self.lowcut_sos = butter(
            N=lowcut_order,
            Wn=lowcut_freq,
            btype='highpass',
            fs=self.fs,
            output='sos')

    def get_frequency_response(self, frequencies: np.ndarray) -> np.ndarray:
        _, highcut_h = sosfreqz(self.highcut_sos, worN=frequencies, fs=self.fs)
        _, lowcut_h = sosfreqz(self.lowcut_sos, worN=frequencies, fs=self.fs)

        return np.abs(highcut_h * lowcut_h)


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
            raise ValueError(
                f"Unsupported type: {bandpass_type}. "
                f"Available: {list(cls._types.keys())}"
            )

        # Instanziieren mit benötigten Parametern
        constructor_args = {
            'butterworth': {
                'fs': config['fs'],
                'lowcut_freq': config['bandpass_params']['lowcut_freq'],
                'highcut_freq': config['bandpass_params']['highcut_freq'],
                'lowcut_order':
                    config['bandpass_params'].get('lowcut_order', 4),
                'highcut_order':
                    config['bandpass_params'].get('highcut_order', 2)
            },
            'null': {}
        }

        bandpass = bandpass_class(**constructor_args.get(bandpass_type, {}))

        return bandpass
