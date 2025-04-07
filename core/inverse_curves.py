"""
Berechnung des inversen Amplitudengangs mit Regularisierung

Klassen:
- ICurveSmoother: Interface für Smoothing der Messdaten
- NullSmoother: kein Smoothing
- FractionalOctaveSmoother: Smoothing in Oktavbändern (zB 1/3 OKtavbänder)
- ERBSmoother: Smoothing in Equivalent Rectangular Bandwidths
- CurveSmootherFactory: Erzeugt CurveSmoother-Instanzen

- INotchMasker: Interface für Notch-Detection und -Masking
- NullMasker: keine Notch-Detection
- ProminenceNotchMasker: Notch-Detection basiert auf Prominence der Notches
- NotchMaskerFactory: Erzeugt INotchMasker Instanzen

- IInverseCurveCalculator: Interface für Methoden zur Berechnung der inversen Übertragungsfunktion
- SimpleInverseCurveCalculator: Berechnung mit einfacher Regularization
- TikhonovInverseCalculator: Berechnung mit Tikhonov-Regularization
- InverseCurveCalculatorFactory: Erzeugt InverseCurveCalculator-Instanzen

Konzepte:
- Glättung der Messdaten ist psycho-akustisch motiviert: 1/3 Oktavband oder ERB-Skala
- Zur Vermeidung von unerwünschter Frequenzüberhöhung im hochfrequenten Bereich wird Regularisierung genutzt
"""
import numpy as np
import math
from scipy.signal import butter, sosfreqz, find_peaks, peak_widths
from abc import ABC, abstractmethod


# ---------------------------- Curve Smoothing ----------------------------
class ICurveSmoother(ABC):
    """Abstrakte Schnittstelle für die Glättung einer Impulsantwort"""

    @abstractmethod
    def smooth(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """Glättet den Amplitudengang einer Impulsantwort"""
        pass


class NullSmoother(ICurveSmoother):
    """Keine Glättung - direkte Verwendung der inversen Antwort"""
    def __init__(self):
        pass

    def smooth(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        return magnitudes  # Keine Modifikation


class FractionalOctaveSmoother(ICurveSmoother):
    def __init__(self, fraction: int = 3):
        self.fraction = fraction

    def smooth(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """ Führt eine fraktionale Oktavbandglättung durch (z. B. 1/3-Oktave) """

        smoothed_magnitudes = np.zeros_like(magnitudes)

        # Berechnung der Grenzfrequenzen
        fd = np.sqrt(2) ** (1 / self.fraction)

        # Smoothing
        for i, f_center in enumerate(frequencies):
            # Indizes im aktuellen Band berechnen
            f_upper = f_center * fd
            f_lower = f_center / fd
            mask = (frequencies >= f_lower) & (frequencies <= f_upper)
            smoothed_magnitudes[i] = np.mean(magnitudes[mask])

        return smoothed_magnitudes


class ERBSmoother(ICurveSmoother):
    """Equivalent Rectangular Bandwidth Glättung nach Moore & Glasberg"""

    def __int__(self):
        pass

    def smooth(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        erb_bandwidths = 24.7 * ((4.37 * frequencies/1000) + 1)
        smoothed_magnitudes = np.zeros_like(magnitudes)

        for i, (f_center, bw) in enumerate(zip(frequencies, erb_bandwidths)):
            mask = (frequencies >= f_center - bw/2) & (frequencies <= f_center + bw/2)
            smoothed_magnitudes[i] = np.mean(magnitudes[mask])

        return smoothed_magnitudes


class CurveSmootherFactory:
    """ Erzeugt den passenden Smoother für den gegebenen Tag"""
    _types = {
        'null': NullSmoother,
        "octave": FractionalOctaveSmoother,
        "erb": ERBSmoother
    }

    @classmethod
    def get_smoother(cls, config) -> ICurveSmoother:

        smoother_type = config['smoothing_type'].lower()
        smoother_class = cls._types.get(smoother_type)
        if not smoother_class:
            raise ValueError(f"Unsupported type: {smoother_type}. Available: {list(cls._types.keys())}")

        # Instanziieren mit benötigten Parametern
        constructor_args = {
            'octave': {'fraction': config.get('smoothing_params').get('fraction')},
            'erb': {},
            'null': {}
        }

        smoother = smoother_class(**constructor_args.get(smoother_type, {}))

        return smoother

# ---------------------------- Notch Masking ----------------------------

class INotchMasker(ABC):
    """ Abstrakte Schnittstelle für Notch-Detection und -Masking"""
    @abstractmethod
    def apply_notch_mask(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        pass


class NullMasker(INotchMasker):
    """ Konkrete Implementierung für 'keine Notch-Detection' """
    def __int__(self):
        pass

    def apply_notch_mask(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        pass


class PromNotchMasker(INotchMasker):
    """ Automatische Notch-Detection und -Masking basierend auf Prominenzanalyse der gemessenen Übertragungsfunktion"""
    def __init__(self, attenuation_db: float=10.0, min_depth_db: float=6.0, prominence: float=3.0, rel_height: float=0.5, smooth_fraction: int=12):
        self.attenuation_db = attenuation_db
        self.min_depth_db = min_depth_db
        self.prominence = prominence
        self.rel_height = rel_height
        self.smooth_fraction = smooth_fraction

    def _detect_notches(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """Notch-Detection basierend auf Prominenzanalyse"""

        # 1. Glättung
        _smoother = CurveSmootherFactory.get_smoother(
            {
                'smoothing_type': 'octave',
                'smoothing_params':{'fraction': self.smooth_fraction}
            })

        smoothed_db = _smoother.smooth(frequencies, magnitudes)

        # 2. Berechnung der Tiefe relativ zur geglätteten Kurve
        depth = smoothed_db - magnitudes  # Positive Werte = Notches

        # 3. Finde Notches mit ausreichender Tiefe und Prominenz
        notches, props = find_peaks(
            depth,
            height=self.min_depth_db,
            prominence=self.prominence,
            width=1,                        # Mindestbreite: 1 Frequencybin
            rel_height=self.rel_height
        )

        # 4. Bestimme Maskierungsbereiche
        mask = np.ones_like(frequencies, dtype=bool)

        for i, f_center in enumerate(notches):
            start = max(0, math.floor(props['left_ips'][i]))
            end = min(len(frequencies), math.ceil(props['right_ips'][i]))
            mask[start:end] = False

        return mask

    def apply_notch_mask(self,
                         frequencies: np.ndarray,
                         magnitudes: np.ndarray,
                         ) -> np.ndarray:
        """ """
        mask = self._detect_notches(frequencies, magnitudes)

        # Verstärkung in den Notch-Bereichen (= Auffüllen der Notches)
        gain = 10 ** (-self.attenuation_db / 20)
        masked_magnitudes = np.where(mask, magnitudes, magnitudes * gain)
        return masked_magnitudes


class NotchMaskerFactory:
    """Erzeugt den passenden Bandpass für gegebenen Typ"""

    _types = {
        'null': NullMasker,
        'prominence': PromNotchMasker
    }

    @classmethod
    def get_masker(cls, config) -> INotchMasker:
        masker_type = config['notch_masking_type'].lower()
        masker_class = cls._types.get(masker_type)
        if not masker_class:
            raise ValueError(f"Unsupported type: {masker_type}. Available: {list(cls._types.keys())}")

        # Instanziieren mit benötigten Parametern
        constructor_args = {
            'prominence': {
                'attenuation_db': config['notch_masking_params']['attenuation_db'],
                'min_depth_db': config['notch_masking_params']['min_depth_db'],
                'prominence': config['notch_masking_params']['prominence'],
                'rel_height': config['notch_masking_params']['rel_height'],
                'smooth_fraction': config['notch_masking_params']['smooth_fraction']
            },
            'null': {}
        }

        masker = masker_class(**constructor_args.get(masker_type, {}))

        return masker
# ---------------------------- Inverse Curve Calculation ----------------------------
class IInverseCurveCalculator(ABC):
    """Abstrakte Schnittstelle für alle Inversionsmethoden"""

    @abstractmethod
    def compute(self,
                frequencies: np.ndarray,
                measured_db: np.ndarray,
                target_mag: np.ndarray,
                **params) -> np.ndarray:
        """Berechnet inversen Amplitudengang"""
        pass


class SimpleInverseCurveCalculator(IInverseCurveCalculator):
    """Berechnet inversen Amplitudengang mit einfacher Regularisierung."""

    def compute(self, frequencies, measured_db, target_mag, **params):
        measured_lin = 10 ** (measured_db / 20)
        return target_mag / (measured_lin + params.get('epsilon', 1e-10))


class TikhonovInverseCalculator(IInverseCurveCalculator):
    """Berechnet inversen Amplitudengang mit Tikhonov-Regularisierung"""

    def compute(self, frequencies, measured_db, target_mag, **params):
        measured_lin = 10 ** (measured_db / 20)
        beta = params.get('beta', 1e-2)

        # Filter für B(w) erstellen
        filter_design = RegularizationFilterFactory.get_filter(params.get('b_filter_type', 'highpass'))
        b_weight = filter_design.get_weight(frequencies, **params)
        regularization = beta * b_weight
        return target_mag / (measured_lin + regularization + 1e-10)


class InverseCurveCalculatorFactory:
    """Factory für Inversionsmethoden"""

    _methods = {
        'simple': SimpleInverseCurveCalculator,
        'tikhonov': TikhonovInverseCalculator,
    }

    @classmethod
    def get_calculator(cls, method: str) -> IInverseCurveCalculator:
        calculator_class = cls._methods.get(method.lower())
        if not calculator_class:
            raise ValueError(f"Unsupported inverse method: {method}. Available: {list(cls._methods.keys())}")
        return calculator_class()

# ---------------------------- Berechnung der Regularisierungsfunktion ----------------------------

class IRegularizationFilterDesign(ABC):
    """Abstrakte Schnittstelle für Regularisierungsfilter"""

    @abstractmethod
    def get_weight(self, frequencies: np.ndarray, **params) -> np.ndarray:
        """Gibt den frequenzabhängigen Gewichtungsvektor zurück"""
        pass


class HighpassRegularizationFilter(IRegularizationFilterDesign):
    """Butterworth-Hochpass für B(w)"""

    def get_weight(self, frequencies, **params):
        sos = butter(
            N=params.get('order', 2),
            Wn=params['cutoff_hz'],
            btype='highpass',
            fs=params['fs'],
            output='sos'
        )
        _, h = sosfreqz(sos, worN=frequencies, fs=params['fs'])
        return np.abs(h)


class LowpassRegularizationFilter(IRegularizationFilterDesign):
    """Butterworth-Tiefpass für B(w)"""

    def get_weight(self, frequencies, **params):
        sos = butter(
            N=params.get('order', 2),
            Wn=params['cutoff_hz'],
            btype='lowpass',
            fs=params['fs'],
            output='sos'
        )
        _, h = sosfreqz(sos, worN=frequencies, fs=params['fs'])
        return np.abs(h)


class RegularizationFilterFactory:
    """Factory für Regularisierungsfilter"""

    _types = {
        'highpass': HighpassRegularizationFilter,
        'lowpass': LowpassRegularizationFilter
    }

    @classmethod
    def get_filter(cls, filter_type: str) -> IRegularizationFilterDesign:
        return cls._types.get(filter_type.lower(), HighpassRegularizationFilter)()