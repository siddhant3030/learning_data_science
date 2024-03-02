import requests

url = f"https://kf.kobotoolbox.org/token/?format=json"
response = requests.get(url, auth=("username", "password"))

json_response = response.json()
print(json_response)

a = json_response.get("token", None), None if json_response is not None else None, None
print(a)
token=a[0]
print(token)
