import requests

a = "https://kf.kobotoolbox.org/api/v2/assets/aLKzTfJAw5KKbtRxLUncwh/data.json"

response = requests.get(a, auth=("username", "password"))
print(response)
json_response = response.json()
print(json_response)