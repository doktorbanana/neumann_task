"""
Target-Kurven - Definition der gewünschten Frequenzcharakteristik

Klassen:
- ITargetCurve: Interface für beliebige Zielkurven
- FlatTargetCurve: Standardimplementierung (lineare Übertragung)
- TargetCurveDesigner: Kombiniert Kurve mit Bandpass-Eigenschaften

Konzepte:
- Ermöglicht nicht-lineare Erweiterungen durch neue
    ITargetCurve-Implementierungen
"""

import numpy as np
from abc import ABC, abstractmethod
from core.bandpass import IBandpassFilter


# ---------------------------- Target Curves ----------------------------

class ITargetCurve(ABC):
    """Abstrakte Schnittstelle für Target-Kurven"""
    @abstractmethod
    def get_curve(self, frequencies: np.ndarray) -> np.ndarray:
        """Gibt die Zielkurve für die gegebenen Frequenzen zurück"""
        pass


class FlatTargetCurve(ITargetCurve):
    """Flache Target-Kurve"""
    def get_curve(self, frequencies: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies)

# --------------------------- Target Curve Designer ---------------------------


class TargetCurveDesigner:
    """Kombiniert Target-Kurve mit Bandpass-Charakteristik"""

    def __init__(self, target_curve: ITargetCurve, bandpass: IBandpassFilter):
        self.bandpass = bandpass
        self.target_curve = target_curve

    def design_curve(self, frequencies: np.ndarray) -> np.ndarray:
        # 1. Target-Kurve generieren
        base_curve = self.target_curve.get_curve(frequencies)

        # 2. Bandpass-Charakteristik anwenden und Kurve normalisieren
        bp_response = self.bandpass.get_frequency_response(frequencies)
        combined = base_curve * bp_response
        return combined / np.max(combined)
