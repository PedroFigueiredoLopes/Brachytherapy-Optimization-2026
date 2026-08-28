import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Callable


class TG43:
    """TG-43 brachytherapy dose calculator with NumPy array support"""
    __slots__ = ['length', 'air_kerma', 'dose_rate_constant', 'radial_func', 'reference_geometry']

    def __init__(self, radial_func: Callable, air_kerma=40000, length = 10e-3, dose_rate_constant = 1.044) -> None:
        self.length = length # Source length in meters
        self.air_kerma = air_kerma  # Air kerma strength (U)
        self.dose_rate_constant = dose_rate_constant
        self.radial_func = radial_func
        self.reference_geometry = self.geometry_function(np.array([0.01]))[0]

    def calculate(self, radii: np.ndarray) -> np.ndarray:
        """
        Calculate dose rate at radii along transverse axis.

        Args:
            radii: Array of distances from source center in meters

        Returns:
            Array of dose rates in cGy/h
        """
        dose_rates = self.air_kerma * self.dose_rate_constant
        dose_rates *= self.geometry_function(radii) / self.reference_geometry
        dose_rates *= self.radial_func(radii)
        return dose_rates

    def geometry_function(self, radii: np.ndarray) -> np.ndarray:
        """Line source geometry function G(r,θ0) for transverse axis (θ=90°)"""
        result = np.pi - 2 * np.arctan(2 * radii / self.length)
        result /= self.length * radii # *1, the anisotropy function is 1
        return result

def load_dose_calculator()->Callable:
    # Load radial dose function data
    file_path = Path(__file__).parent / "192ir-hdr_varianclassic.xls"
    df = pd.read_excel(file_path, sheet_name='Radial Dose Function ', header=3, usecols="C, D")
    df.dropna(inplace=True)
    df.columns = ['r_cm', 'g']
    df['r_m'] = df['r_cm'] / 100  # Convert from centimeters to meters

    # Create interpolation function
    radial_func = interp1d(df['r_m'], df['g'], fill_value='extrapolate')

    # Initialize calculator
    source = TG43(radial_func)

    return source.calculate


def main():
    """Test TG-43 implementation against published data"""

    # Load radial dose function data
    file_path = Path(__file__).parent / "192ir-hdr_varianclassic.xls"
    df = pd.read_excel(file_path, sheet_name='Radial Dose Function ', header=3, usecols="C, D")
    df.dropna(inplace=True)
    df.columns = ['r_cm', 'g']
    df['r_m'] = df['r_cm'] / 100  # Convert from centimeters to meters
    print(df)

    # Create interpolation function
    radial_func = interp1d(df['r_m'], df['g'])

    # Initialize calculator
    source = TG43(radial_func, 1)  # The air kerma of 1 is used to conform with reference data

    # Test points (convert cm to m)
    test_points_cm = np.array([0.10, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 10])
    test_points_m = test_points_cm / 100

    # Calculate doses
    print(source.calculate(0.01))
    calculate_vec = np.vectorize(source.calculate)
    predicted = calculate_vec(test_points_m)
    print(predicted)

    # Published reference data
    reference_doses = np.array(
        [29.418, 9.7037, 3.4958, 1.7563, 1.0432, 0.4842, 0.2777, 0.1793, 0.1247, 0.0705, 0.0447, 0.0307, 0.0221,
         0.0100])
    print(reference_doses)
    print(reference_doses - predicted)


if __name__ == '__main__':
    main()
