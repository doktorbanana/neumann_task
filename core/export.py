"""
Exportkomponenten - Persistierung der Filterkoeffizienten

Klassen:
- IFilterExporter: Interface für Export nach Dateiformat
- WavExporter: 32-bit Float WAV Format
- FilterExporter: Zentrale Exportroutine mit automatischer Formatdetektion

Konzepte:
- Erweiterbar um zusätzliche Exportformate
- Fehlertolerante Dateibehandlung
"""

import numpy as np
from scipy.io import wavfile
from abc import ABC, abstractmethod

# ---------------------------- Exporter ----------------------------


class IFilterExporter(ABC):
    """ Exportiert Filterkoeffizienten als Datei"""

    @staticmethod
    @abstractmethod
    def export(filepath: str, filter_coeffs: np.ndarray, fs: int):
        pass


class WavExporter(IFilterExporter):
    """Exportiert Koeffizienten als Impulsantwort in WAV-Datei"""
    @staticmethod
    def export(file_path: str, filter_coeffs: np.ndarray, fs: int) -> None:
        wavfile.write(file_path, fs, filter_coeffs.astype(np.float32))

# ---------------------------- Export Manager ----------------------------


class FilterExporter:
    """Zentrale Exportklasse mit automatischer Formaterkennung
    (Bei Bedarf können mehr Dateiformate implementiert werden)"""

    _exporters = {
        'wav': WavExporter
    }

    @classmethod
    def export(cls, coeffs: np.ndarray, file_path: str, fs: int) -> None:
        """Exportiert Filterkoeffizienten in das passende Format"""

        file_ext = file_path.split('.')[-1].lower()
        exporter = cls._exporters.get(file_ext)

        if not exporter:
            raise ValueError(f"Unsupported format: {file_ext}. "
                             f"Supported: {list(cls._exporters.keys())}")

        exporter.export(file_path, coeffs, fs)
