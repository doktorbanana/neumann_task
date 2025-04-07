"""
Datenverwaltung - Einlesen und Validierung von Messdaten

Klassen:
- SpectrumData: Datencontainer
- ISpectrumDataLoader: Interface für Ladefunktionen je nach Dateiformat
- DataLoaderFactory: Zentrale Zugriffskomponente

Funktionen:
- Erweiterbar für Unterstützung verschiedener Dateiformate
- Automatische Formatdetektion
"""

import json
import numpy as np
from abc import ABC, abstractmethod


# ---------------------------- Data Handling ----------------------------
class SpectrumData:
    """Datencontainer für Frequenz- und dB-Werte."""

    def __init__(self, frequencies: np.ndarray, db_values: np.ndarray):
        self.frequencies = frequencies
        self.db_values = db_values


class ISpectrumDataLoader(ABC):
    """Abstrakte Schnittstelle für Spectrum Data Loader"""
    @staticmethod
    @abstractmethod
    def supports_format(file_path: str) -> bool:
        """Prüft ob das Format unterstützt wird"""
        pass

    @staticmethod
    @abstractmethod
    def load(file_path: str) -> SpectrumData:
        """Lädt Spektrum aus der Datei"""
        pass


class JsonSpectrumDataLoader(ISpectrumDataLoader):
    """Lädt Daten aus JSON-Dateien"""

    @staticmethod
    def supports_format(file_path: str) -> bool:
        return file_path.lower().endswith('.json')

    @staticmethod
    def load(file_path: str) -> SpectrumData:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return SpectrumData(
                frequencies=np.array(data['f_hz']),
                db_values=np.array(data['db'])
            )
        except Exception as e:
            raise ValueError(f"JSON loading error: {e}")


class DataLoaderFactory:
    """Erzeugt den passenden DataLoader basierend auf der Dateiendung
    (Bei Bedarf können mehr Dateiformate implementiert werden)"""
    _loaders = [JsonSpectrumDataLoader()]

    @classmethod
    def get_loader(cls, file_path: str) -> ISpectrumDataLoader:
        for loader in cls._loaders:
            if loader.supports_format(file_path):
                return loader
        raise ValueError(f"Unsupported file format: {file_path}")
