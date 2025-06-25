# src/core/analysis/danger_calculator.py

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
from sklearn.naive_bayes import GaussianNB

from config.settings import get_settings
from src.core.analysis.gpt_analyzer import analyze_image
from src.core.analysis.weather_service import get_weather

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration settings
settings = get_settings()


class DangerCalculator:
    """
    Calculate danger levels based on multiple factors including GPT analysis,
    weather conditions, and fall duration.
    """

    def __init__(self, bayesian_model: Optional[GaussianNB] = None):
        """
        Initialize the danger calculator.

        Args:
            bayesian_model: Optional pre-trained Bayesian model for predictions
        """
        self.bayesian_model = bayesian_model
        self.danger_settings = settings.DANGER_SETTINGS

        # Default factor weights (can be adjusted based on requirements)
        self.default_weights = {
            "age": 0.4,
            "weather": 0.3,
            "time_of_day": 0.2,
            "fall_duration": 0.5,
            "environment": 0.3,
            "lighting": 0.2
        }

        logger.info("DangerCalculator initialized")

    def train_bayesian_model(self, training_data: np.ndarray, labels: np.ndarray) -> GaussianNB:
        """
        Train the Bayesian model for danger prediction.

        Args:
            training_data: Training data (numpy array)
            labels: Corresponding danger level labels

        Returns:
            Trained Bayesian model
        """
        try:
            model = GaussianNB()
            model.fit(training_data, labels)
            self.bayesian_model = model
            logger.info("Bayesian model trained successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to train Bayesian model: {e}")
            raise

    def map_gpt_analysis_to_factors(self,
                                    analysis_result: Dict[str, Any],
                                    weather_code: Optional[int] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Map GPT analysis and weather data to danger factors.

        Args:
            analysis_result: JSON result from analyze_image
            weather_code: Weather code from get_weather (optional)

        Returns:
            Tuple of (factors, weights) dictionaries
        """
        gpt_analysis = analysis_result.get('gpt_analysis', {})

        # Map analysis results to danger factors (0.0 to 1.0 scale)
        factors = {
            "age": self._calculate_age_factor(gpt_analysis.get('possible_age', 'adults')),
            "weather": self._calculate_weather_factor(weather_code),
            "time_of_day": self._calculate_time_factor(gpt_analysis.get('time_of_day', 'day')),
            "environment": self._calculate_environment_factor(gpt_analysis.get('environment', 'indoor')),
            "lighting": self._calculate_lighting_factor(gpt_analysis.get('lighting', 'bright')),
            "status": self._calculate_status_factor(gpt_analysis.get('status', [])),
            "face_to_ground": self._calculate_face_factor(gpt_analysis.get('faceToTheGround', 'false'))
        }

        # Use default weights (can be customized based on scenario)
        weights = self.default_weights.copy()

        # Adjust weights based on detected conditions
        if 'fall' in gpt_analysis.get('status', []):
            weights['fall_duration'] = 0.6  # Increase importance if fall detected

        if gpt_analysis.get('environment') in ['road', 'water']:
            weights['environment'] = 0.4  # Increase importance for dangerous environments

        logger.debug(f"Calculated factors: {factors}")
        logger.debug(f"Applied weights: {weights}")

        return factors, weights

    def _calculate_age_factor(self, age_group: str) -> float:
        """Calculate danger factor based on age group."""
        age_factors = {
            'old_people': 0.9,  # Higher risk for elderly
            'adults': 0.6,  # Medium risk for adults
            'children': 0.8  # Higher risk for children
        }
        return age_factors.get(age_group.lower(), 0.6)

    def _calculate_weather_factor(self, weather_code: Optional[int]) -> float:
        """Calculate danger factor based on weather conditions."""
        if weather_code is None:
            return 0.5  # Default neutral weather

        # Weather codes: 0-3 clear/good, higher values indicate worse conditions
        if weather_code <= 3:
            return 0.3  # Good weather, lower danger
        elif weather_code <= 50:
            return 0.6  # Moderate weather
        else:
            return 0.9  # Bad weather, higher danger

    def _calculate_time_factor(self, time_of_day: str) -> float:
        """Calculate danger factor based on time of day."""
        time_factors = {
            'day': 0.4,  # Lower risk during day
            'night': 0.7  # Higher risk at night
        }
        return time_factors.get(time_of_day.lower(), 0.5)

    def _calculate_environment_factor(self, environment: str) -> float:
        """Calculate danger factor based on environment."""
        environment_factors = {
            'indoor': 0.3,  # Safer indoors
            'road': 0.9,  # Very dangerous on road
            'water': 1.0,  # Extremely dangerous in water
            'blaze': 1.0,  # Extremely dangerous in fire
            'bed': 0.2,  # Very safe in bed
            'chair': 0.3  # Safe on chair/furniture
        }
        return environment_factors.get(environment.lower(), 0.5)

    def _calculate_lighting_factor(self, lighting: str) -> float:
        """Calculate danger factor based on lighting conditions."""
        lighting_factors = {
            'bright': 0.2,  # Good visibility, lower risk
            'dim': 0.6,  # Poor visibility, higher risk
            'dark': 0.8  # Very poor visibility, high risk
        }
        return lighting_factors.get(lighting.lower(), 0.5)

    def _calculate_status_factor(self, status_list: list) -> float:
        """Calculate danger factor based on detected status conditions."""
        if not isinstance(status_list, list):
            return 0.5

        # High-risk status conditions
        high_risk_statuses = ['bleeding', 'accident', 'pain', 'hurt', 'drowning', 'fall']
        medium_risk_statuses = ['sit']
        low_risk_statuses = ['walk']

        max_risk = 0.0
        for status in status_list:
            if status.lower() in high_risk_statuses:
                max_risk = max(max_risk, 0.9)
            elif status.lower() in medium_risk_statuses:
                max_risk = max(max_risk, 0.5)
            elif status.lower() in low_risk_statuses:
                max_risk = max(max_risk, 0.2)

        return max_risk if max_risk > 0 else 0.5

    def _calculate_face_factor(self, face_to_ground: str) -> float:
        """Calculate danger factor based on face orientation."""
        return 0.8 if face_to_ground.lower() == 'true' else 0.3

    def evaluate_danger_factors(self,
                                factors: Dict[str, float],
                                weights: Dict[str, float],
                                nonlinear_amplification: float = 2.0) -> float:
        """
        Evaluate danger value based on factors and weights.

        Args:
            factors: Dictionary of danger factors (0.0 to 1.0)
            weights: Dictionary of weights for each factor
            nonlinear_amplification: Nonlinear amplification factor

        Returns:
            Final danger value (0.0 to 1.0)
        """
        try:
            # Ensure factors and weights have matching keys
            common_keys = set(factors.keys()) & set(weights.keys())
            if not common_keys:
                logger.warning("No common keys between factors and weights")
                return 0.5

            # Calculate weighted danger score
            weighted_sum = sum(factors[key] * weights[key] for key in common_keys)
            total_weight = sum(weights[key] for key in common_keys)

            if total_weight == 0:
                logger.warning("Total weight is zero")
                return 0.5

            danger_score = weighted_sum / total_weight

            # Apply nonlinear amplification
            danger_score = danger_score ** nonlinear_amplification

            # Use Bayesian model if available
            if self.bayesian_model:
                try:
                    factor_array = np.array([list(factors.values())]).reshape(1, -1)
                    bayesian_prediction = self.bayesian_model.predict_proba(factor_array)[0]
                    # Combine rule-based and ML-based predictions
                    danger_score = 0.7 * danger_score + 0.3 * bayesian_prediction[1]
                except Exception as e:
                    logger.warning(f"Bayesian prediction failed: {e}")

            # Ensure result is within bounds
            return max(0.0, min(1.0, danger_score))

        except Exception as e:
            logger.error(f"Error evaluating danger factors: {e}")
            return 0.5  # Return neutral danger level on error

    def calculate_lec_risk(self,
                           danger_value: float,
                           falling_duration: int,
                           likelihood: float = 1.0) -> float:
        """
        Calculate risk score using LEC (Likelihood, Exposure, Consequence) model.

        Args:
            danger_value: Calculated danger value (0.0 to 1.0)
            falling_duration: Duration of the fall event in frames/seconds
            likelihood: Likelihood of the event (default is 1.0)

        Returns:
            LEC risk score
        """
        try:
            # Normalize falling duration (convert to a 0-1 scale)
            max_duration = 300  # Maximum expected duration (5 minutes at 60fps)
            normalized_duration = min(falling_duration / max_duration, 1.0)

            lec_score = likelihood * normalized_duration * danger_value

            logger.debug(
                f"LEC calculation: L={likelihood}, E={normalized_duration}, C={danger_value}, Score={lec_score}")

            return lec_score

        except Exception as e:
            logger.error(f"Error calculating LEC risk: {e}")
            return 0.5

    def calculate_danger_and_lec(self,
                                 image_path: str,
                                 weather_data: Optional[Tuple] = None,
                                 falling_duration: int = 0) -> Tuple[float, float]:
        """
        Calculate both danger value and LEC risk score for a given image.

        Args:
            image_path: Path to the image file
            weather_data: Optional tuple containing (temperature, weather_code, description)
            falling_duration: Duration of the fall event

        Returns:
            Tuple of (danger_value, lec_risk_score)
        """
        try:
            # Analyze image with GPT
            analysis_result = analyze_image(image_path)

            # Extract weather information
            weather_code = weather_data[1] if weather_data else None

            # Calculate danger factors
            factors, weights = self.map_gpt_analysis_to_factors(analysis_result, weather_code)

            # Evaluate danger
            danger_value = self.evaluate_danger_factors(factors, weights)

            # Calculate LEC risk
            lec_risk_score = self.calculate_lec_risk(danger_value, falling_duration)

            logger.info(f"Danger calculation completed: danger={danger_value:.2f}, lec_risk={lec_risk_score:.2f}")

            return danger_value, lec_risk_score

        except Exception as e:
            logger.error(f"Error in danger calculation: {e}")
            return 0.5, 0.5  # Return neutral values on error

    def calculate_danger(self,
                         analysis_result: Dict[str, Any],
                         falling_duration: int,
                         weather_data: Optional[Tuple] = None) -> float:
        """
        Calculate danger value based on analysis results and falling duration.

        Args:
            analysis_result: Analysis JSON result from the GPT model
            falling_duration: Duration of the fall event
            weather_data: Optional weather data (temperature, weather_code)

        Returns:
            Danger value (0.0 to 1.0)
        """
        try:
            weather_code = weather_data[1] if weather_data else None

            # Map analysis to factors
            factors, weights = self.map_gpt_analysis_to_factors(analysis_result, weather_code)

            # Add fall duration factor
            max_duration = self.danger_settings.get('fall_duration_alert', 5) * 60  # Convert to frames
            duration_factor = min(falling_duration / max_duration, 1.0)
            factors['fall_duration'] = duration_factor

            # Evaluate danger
            danger_value = self.evaluate_danger_factors(factors, weights)

            logger.debug(f"Calculated danger value: {danger_value:.2f}")

            return danger_value

        except Exception as e:
            logger.error(f"Error calculating danger: {e}")
            return 0.5

    def save_calculation_results(self,
                                 results: Dict[str, Any],
                                 output_dir: Optional[str] = None):
        """
        Save danger calculation results to file.

        Args:
            results: Dictionary containing calculation results
            output_dir: Output directory (defaults to outputs/evaluation_results/)
        """
        try:
            if output_dir is None:
                output_dir = settings.EVALUATION_RESULTS_DIR

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"danger_calculation_{timestamp}.json"

            output_file = output_path / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Danger calculation results saved to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to save calculation results: {e}")


# Global instance for easy access
danger_calculator = DangerCalculator()


def calculate_danger(analysis_result: Dict[str, Any],
                     falling_duration: int,
                     weather_data: Optional[Tuple] = None) -> float:
    """
    Convenience function for calculating danger using global instance.

    Args:
        analysis_result: Analysis JSON result from GPT model
        falling_duration: Duration of the fall event
        weather_data: Optional weather data

    Returns:
        Danger value (0.0 to 1.0)
    """
    return danger_calculator.calculate_danger(analysis_result, falling_duration, weather_data)


# Example usage and testing
if __name__ == "__main__":
    # Example training data for Bayesian model
    training_data = np.array([
        [0.8, 0.7, 0.5, 0.9, 0.6, 0.8, 0.7],  # High risk scenario
        [0.6, 0.5, 0.3, 0.8, 0.4, 0.5, 0.4],  # Medium risk scenario
        [0.4, 0.2, 0.7, 0.6, 0.2, 0.3, 0.2],  # Low risk scenario
    ])
    labels = np.array([1, 1, 0])  # 1 = high danger, 0 = low danger

    # Initialize calculator and train model
    calculator = DangerCalculator()
    calculator.train_bayesian_model(training_data, labels)

    # Test with sample data
    sample_analysis = {
        'gpt_analysis': {
            'possible_age': 'old_people',
            'time_of_day': 'night',
            'environment': 'road',
            'lighting': 'dark',
            'status': ['fall', 'pain'],
            'faceToTheGround': 'true'
        }
    }

    sample_weather = (15, 60, 'rainy')  # temperature, weather_code, description
    sample_duration = 180  # 3 minutes

    try:
        danger_value, lec_risk = calculator.calculate_danger_and_lec(
            "test_image.jpg",  # Would need actual image
            sample_weather,
            sample_duration
        )

        print(f"Calculated Danger Value: {danger_value:.2f}")
        print(f"LEC Risk Score: {lec_risk:.2f}")

    except Exception as e:
        print(f"Test failed: {e}")

    # Test simple danger calculation
    danger_value = calculator.calculate_danger(sample_analysis, sample_duration, sample_weather)
    print(f"Simple Danger Value: {danger_value:.2f}")