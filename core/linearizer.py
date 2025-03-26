"""
Linearizer-Klasse - Zentrale Steuerung der Frequenzganglinearisierung

Funktionen:
- Orchestriert den gesamten Equalization-Prozess
- Koordiniert Datenfluss zwischen Komponenten
- Implementiert die Hauptpipeline:
  1. Daten laden
  2. Zielkurve generieren
  3. Inverse Filterantwort berechnen
  4. Filterkoeffizienten erzeugen
  5. Ergebnisse visualisieren/exportieren
"""

import numpy as np
from scipy.signal import freqz
from core.bandpass import BandpassFactory
from core.data_handling import DataLoaderFactory
from core.target_curves import FlatTargetCurve, TargetCurveDesigner
from core.filter_design import InverseCurveCalculator, FilterDesignerFactory
from core.visualization import ResponsePlotter
from core.export import FilterExporter


class Linearizer:
    """Orchestriert den Prozess der Linearisierung des Frequenzgangs"""

    def __init__(self, config):
        self.config = config
        self.data = None
        self.target_mag = None
        self.filter_coeffs = None
        self.bandpass = BandpassFactory.get_bandpass(self.config)

        self._validate_config()

    def _validate_config(self):
        required_keys = ['fs', 'fir_taps', 'regularization', 'design_method', 'bandpass_type']

        missing = [element for element in required_keys if element not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        if self.config['bandpass_type'] == 'butterworth' and 'order' not in self.config:
            raise ValueError("Butterworth filter requires 'order' parameter")

    def load_data(self, file_path):
        loader = DataLoaderFactory.get_loader(file_path)
        self.data = loader.load(file_path)

    def design_target_curve(self):
        flat_curve = FlatTargetCurve()
        target_designer = TargetCurveDesigner(bandpass=self.bandpass, target_curve=flat_curve)
        self.target_mag = target_designer.design_curve(self.data.frequencies)

    def calculate_inverse_response(self):
        return InverseCurveCalculator.compute(
            self.data.db_values, self.target_mag, self.config['regularization']
        )

    def design_filter(self, inverse_response):
        designer = FilterDesignerFactory.get_designer(self.config['design_method'])
        self.filter_coeffs = designer.design(
            self.data.frequencies,
            inverse_response,
            self.config['fs'],
            self.config['fir_taps']
        )

    def simulate_response(self):
        w, h = freqz(self.filter_coeffs, worN=self.data.frequencies, fs=self.config['fs'])
        return 20 * np.log10(np.abs(h)) + self.data.db_values

    def plot_results(self, equalized_db):
        ResponsePlotter.plot(
            self.data.frequencies,
            self.data.db_values,
            20 * np.log10(self.target_mag + 1e-10),
            equalized_db
        )

    def export_results(self, filepath):
        FilterExporter.export(self.filter_coeffs, filepath, fs=self.config['fs'])
