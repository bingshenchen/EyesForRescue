import numpy as np
from sklearn.naive_bayes import GaussianNB
from getweer import get_weather
from poging_gen import analyze_image


# Initialize Bayesian model
def train_bayesian_model(training_data, labels):
    """
    Train the Bayesian model.
    :param training_data: Training data (numpy array)
    :param labels: Corresponding danger level labels for the data
    :return: Trained Bayesian model
    """
    model = GaussianNB()
    model.fit(training_data, labels)
    return model


def map_gpt_analysis_to_factors(analysis_result, weather_code):
    """
    Map gpt_analysis and weather data to danger factors.
    :param analysis_result: JSON result from analyze_image
    :param weather_code: Weather code from get_weather
    :return: dict of factors and weights
    """
    factors = {
        "age": 0.8 if analysis_result.get('possible_age') == 'adults' else \
                0.9 if analysis_result.get('possible_age') == 'old_people' else 0.6,
        "weather": 0.7 if weather_code < 3 else 0.4,
        "time_of_day": 0.5 if analysis_result.get('time_of_day') == 'day' else 0.3,
        "fall_duration": 0.9 if 'fall' in analysis_result.get('status', []) else 0.5,
    }
    weights = {
        "age": 0.4,
        "weather": 0.3,
        "time_of_day": 0.2,
        "fall_duration": 0.5,
    }
    return factors, weights


def evaluate_danger_factors(factors, weights, n=2, bayesian_model=None):
    """
    Evaluate danger value.
    :param factors: dict, Scores for each danger factor
    :param weights: dict, Weights for each factor
    :param n: int, Nonlinear amplification factor
    :param bayesian_model: Trained Bayesian model (optional)
    :return: float, Final danger value
    """
    assert set(factors.keys()) == set(weights.keys()), "Factors and weights must have the same keys."
    danger_score = sum(factors[key] * weights[key] for key in factors.keys())
    danger_score = danger_score ** n

    if bayesian_model:
        factor_array = np.array([list(factors.values())])
        bayesian_prediction = bayesian_model.predict_proba(factor_array)[0]
        danger_score = 0.7 * danger_score + 0.3 * bayesian_prediction[1]

    return danger_score


def calculate_lec_risk(danger_value, falling_durations, likelihood=1):
    """
    Calculate risk score using LEC model.
    :param danger_value: Calculated danger value
    :param falling_durations: Duration of the fall event
    :param likelihood: Likelihood of the event (default is 1)
    :return: float, LEC risk score
    """
    lec_score = likelihood * falling_durations * danger_value
    return lec_score


def calculate_danger_and_lec(image_path, weather_data, falling_durations, bayesian_model):
    """
    Calculate the danger value and LEC risk score for a given image and weather data.
    :param image_path: Path to the image file
    :param weather_data: Tuple containing temperature, weather code, and description
    :param falling_durations: Duration of the fall event
    :param bayesian_model: Trained Bayesian model
    :return: tuple of (danger_value, lec_risk_score)
    """
    analysis_result = analyze_image(image_path)
    temperature, weather_code, _ = weather_data

    factors, weights = map_gpt_analysis_to_factors(analysis_result, weather_code)
    danger_value = evaluate_danger_factors(factors, weights, n=2, bayesian_model=bayesian_model)
    lec_risk_score = calculate_lec_risk(danger_value, falling_durations)

    return danger_value, lec_risk_score


def calculate_danger(analysis_result, falling_durations, weather_data=None, bayesian_model=None):
    """
    Calculate the danger value based on analysis results and falling durations.
    :param analysis_result: Analysis JSON result from the GPT model.
    :param falling_durations: Duration of the fall event.
    :param weather_data: Optional weather data (temperature, weather code).
    :param bayesian_model: Trained Bayesian model for additional predictions.
    :return: Danger value.
    """
    # Example factors mapping
    weather_code = weather_data[1] if weather_data else 0
    factors, weights = map_gpt_analysis_to_factors(analysis_result, weather_code)

    # Evaluate danger
    danger_value = evaluate_danger_factors(factors, weights, n=2, bayesian_model=bayesian_model)

    # Example LEC risk calculation (optional)
    lec_risk_score = calculate_lec_risk(danger_value, falling_durations)

    return danger_value


# Example usage
def main():
    training_data = np.array([
        [0.8, 0.7, 0.5, 0.9],
        [0.6, 0.5, 0.3, 0.8],
        [0.4, 0.2, 0.7, 0.6],
    ])
    labels = np.array([1, 0, 0])

    bayesian_model = train_bayesian_model(training_data, labels)

    latitude, longitude = 50.980885, 5.058031
    weather_data = get_weather(latitude, longitude)

    image_path = r"C:\\Users\\Bingshen\\Pictures\\AI Train\\f.jpg"
    falling_durations = 150

    danger_value, lec_risk_score = calculate_danger_and_lec(
        image_path, weather_data, falling_durations, bayesian_model
    )

    print(f"Calculated Danger Value: {danger_value:.2f}")
    print(f"LEC Risk Score: {lec_risk_score:.2f}")


if __name__ == "__main__":
    main()
