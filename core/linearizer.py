"""
Linearizer-Klasse - Zentrale Steuerung der Frequenzganglinearisierung

Funktionen:
- Orchestriert den gesamten Equalization-Prozess
- Koordiniert Datenfluss zwischen Komponenten
- Implementiert die Hauptpipeline:
  1. Daten laden und glätten
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
from core.inverse_curves import CurveSmootherFactory, InverseCurveCalculatorFactory, NotchMaskerFactory
from core.filter_design import FilterDesignerFactory
from core.visualization import ResponsePlotter
from core.export import FilterExporter


class Linearizer:
    """Orchestriert den Prozess der Linearisierung des Frequenzgangs"""

    def __init__(self, config):
        self.config = config
        self.data = None
        self.smoothed_db_values = None
        self.target_filter_response_db = None
        self.filter_coeffs = None
        self.bandpass = BandpassFactory.get_bandpass(self.config)
        self.smoother = CurveSmootherFactory.get_smoother(self.config)
        self.inverse_curve_calculator = InverseCurveCalculatorFactory.get_calculator(self.config.get('inverse_method', 'tikhonov'))
        self.masker = NotchMaskerFactory.get_masker(self.config)

        self._validate_config()

    def _validate_config(self):
        required_keys = ['fs',
                         'fir_taps',
                         'design_method',
                         'bandpass_type',
                         'smoothing_type',
                         'inverse_method',
                         'notch_masking_type']

        missing = [element for element in required_keys if element not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def load_data(self, file_path):
        loader = DataLoaderFactory.get_loader(file_path)
        self.data = loader.load(file_path)

    def mask_notches(self):
        if self.target_filter_response_db is None:
            self.target_filter_response_db = self.masker.apply_notch_mask(
                self.data.frequencies,
                self.data.db_values
            )
        else:
            self.target_filter_response_db = self.masker.apply_notch_mask(
                self.data.frequencies,
                self.target_filter_response_db
            )

    def smooth_data(self):
        if self.target_filter_response_db is None:
            self.target_filter_response_db = self.smoother.smooth(self.data.frequencies, self.data.db_values)
        else:
            self.target_filter_response_db = self.smoother.smooth(self.data.frequencies, self.target_filter_response_db)

        self.smoothed_db_values = self.smoother.smooth(self.data.frequencies, self.data.db_values)


    def design_target_curve(self):
        flat_curve = FlatTargetCurve()
        target_designer = TargetCurveDesigner(bandpass=self.bandpass, target_curve=flat_curve)
        self.target_mag = target_designer.design_curve(self.data.frequencies)

    def calculate_inverse_response(self):

        if self.data is None:
            ValueError("Data must be loaded first!")

        if self.target_filter_response_db is not None:
            inverse_response = self.inverse_curve_calculator.compute(
                self.data.frequencies,
                self.target_filter_response_db,
                self.target_mag,
                **self.config.get('inverse_params', {})
            )
        else:
            inverse_response = self.inverse_curve_calculator.compute(
                self.data.frequencies,
                self.data.db_values,
                self.target_mag,
                **self.config.get('inverse_params', {})
            )

        self.target_filter_response_db = 20 * np.log10(inverse_response)

        return inverse_response

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

        if self.target_filter_response_db is not None:
            result = 20 * np.log10(np.abs(h)) - self.target_filter_response_db
        else:
            result = 20 * np.log10(np.abs(h)) + self.data.db_values
        return result

    def plot_results(self, equalized_db):
        ResponsePlotter.plot(
            self.data.frequencies,
            self.data.db_values,
            self.smoothed_db_values,
            self.target_filter_response_db,
            equalized_db
        )

    def export_results(self, filepath):
        FilterExporter.export(self.filter_coeffs, filepath, fs=self.config['fs'])
