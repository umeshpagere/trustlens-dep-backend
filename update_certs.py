import urllib.request
import certifi
import ssl

context = ssl.create_default_context(cafile=certifi.where())
urllib.request.urlopen('https://www.google.com', context=context)
print("✅ Certifi context successful")
