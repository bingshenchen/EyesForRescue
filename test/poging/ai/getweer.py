import requests
from src.demoForKlant.gps import getLoc


def get_weather(latitude, longitude):
    """
    Get weather information using the Open-Meteo API.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    print(data)
    if 'current_weather' in data:
        temperature = data['current_weather']['temperature']
        weather_code = data['current_weather']['weathercode']
        time = data['current_weather']['time']
        return temperature, weather_code, time
    else:
        raise Exception("Error fetching weather data")


if __name__ == "__main__":
    latitude, longitude = getLoc()
    temperature, weather_code, time = get_weather(latitude, longitude)
    print(f"Current temperature: {temperature}°C, Weather code: {weather_code}")
