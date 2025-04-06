"""
Filterdesign - Generierung der FIR-Filterkoeffizienten

Klassen:
- InverseFilterCalculator: Berechnung der inversen Übertragungsfunktion
- IFilterDesigner: Interface für Filterdesign-Algorithmen
- FirlsDesigner: Least-Squares-Optimierte Methode
- Firwin2Designer: Fensterbasierte Methode
- FilterDesignerFactory: Zentraler Zugriff auf Designmethoden

Konzepte:
- Firls: Minimiert mittlere quadratische Abweichung. Liefert bessere Ergebnisse bei höherer Rechenlast.
- Firwin2: Ideale Filterantwort + Fensterung
- Regularisierung verhindert Division durch null
- Da der Filter offline erstellt wird und nicht in real-time angepasst werden muss, wird die Firls-Methode verwendet.
"""

import numpy as np
from scipy.signal import firls, firwin2
from abc import ABC, abstractmethod


# ---------------------------- Inverse Curve Calculation ----------------------------
class InverseCurveCalculator:
    """Berechnet die inverse Filterantwort mit Regularisierung."""

    @staticmethod
    def compute(measured_db, target_mag, regularization=1e-2):
        measured_lin = 10 ** (measured_db / 20)
        return target_mag / (measured_lin + regularization)


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


# ---------------------------- Filter Designer ----------------------------
class IFilterDesigner(ABC):
    """Abstrakte Schnittstelle für Filterdesigner"""

    @abstractmethod
    def design(self, frequencies: np.ndarray, response: np.ndarray, fs: int, numtaps: int) -> np.ndarray:
        """ Erzeugt Filterkoeffizienten"""
        pass


class FirlsDesigner(IFilterDesigner):
    """FIR-Filterdesign mit Least Squared Error Methode"""

    def design(self, frequencies: np.ndarray, response: np.ndarray, fs: int, numtaps: int) -> np.ndarray:
        nyq = 0.5 * fs
        normalized_freq = frequencies / nyq
        bands = np.repeat(frequencies / nyq, 2)[1:-1]
        desired = np.interp(
            bands,
            normalized_freq,
            response,
            left=response[0],
            right=response[-1]
        )
        return firls(numtaps, bands, desired)


class Firwin2Designer(IFilterDesigner):
    """FIR-Filterdesign mit firwin2-Methode"""

    def design(self, frequencies, response, fs, numtaps):
        nyq = 0.5 * fs
        norm_freq = np.array(frequencies) / nyq  # Normalisierte Frequenzen
        norm_freq = np.insert(norm_freq, 0, 0.0)
        norm_freq = np.insert(norm_freq, np.size(norm_freq), 1.0)
        response = np.insert(response, 0, 0.0)
        response = np.insert(response, np.size(response), 0.0)
        return firwin2(numtaps, norm_freq, response)


class FilterDesignerFactory:
    """Erzeugt den passenden Filterdesigner für gegebene Methode"""
    _methods = {
        'firls': FirlsDesigner(),
        'firwin2': Firwin2Designer(),
    }

    @classmethod
    def get_designer(cls, method: str) -> IFilterDesigner:
        designer = cls._methods.get(method.lower())
        if not designer:
            raise ValueError(f"Unsupported method: {method}. Available: {list(cls._methods.keys())}")
        return designer
