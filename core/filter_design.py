"""
Filterdesign - Generierung der FIR-Filterkoeffizienten

Klassen:
- IFilterDesigner: Interface für Filterdesign-Algorithmen
- FirlsDesigner: Least-Squares-Optimierte Methode
- Firwin2Designer: Fensterbasierte Methode
- FilterDesignerFactory: Zentraler Zugriff auf Designmethoden

Konzepte:
- Firls: Minimiert mittlere quadratische Abweichung. Liefert bessere Ergebnisse bei höherer Rechenlast.
- Firwin2: Ideale Filterantwort + Fensterung
- Da der Filter offline erstellt wird und nicht in real-time angepasst werden muss, wird die Firls-Methode verwendet.
"""

import numpy as np
from scipy.signal import firls, firwin2
from abc import ABC, abstractmethod


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
