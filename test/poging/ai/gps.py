import requests
import asyncio
import winsdk.windows.devices.geolocation as wdg


async def getCoords():
    locator = wdg.Geolocator()
    pos = await locator.get_geoposition_async()
    return [pos.coordinate.latitude, pos.coordinate.longitude]


def getLoc():
    try:
        return asyncio.run(getCoords())
    except PermissionError:
        print("ERROR: You need to allow applications to access your location in Windows settings")


def get_address(latitude, longitude):
    """
    Use the Nominatim Reverse Geocoding API to retrieve address information.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"
    headers = {
        'User-Agent': 'Python-Geolocation-Script'  # Required User-Agent to avoid request rejection
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        address = data.get('display_name', 'Unknown address')
        return address
    else:
        raise Exception(f"Error fetching address: HTTP {response.status_code}")


if __name__ == "__main__":
    # Get latitude and longitude
    latitude, longitude = getLoc()
    print(f"Longitude: {longitude}, Latitude: {latitude}")

    # Get detailed address
    try:
        address = get_address(latitude, longitude)
        print(f"Detailed Address: {address}")
    except Exception as e:
        print(e)
