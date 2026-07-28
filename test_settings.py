from app.config.settings import settings

print("City:", settings.DEFAULT_CITY)
print("Latitude:", settings.LATITUDE)
print("Longitude:", settings.LONGITUDE)
print("API Key Loaded:", settings.OPENWEATHER_API_KEY[:8] + "...")