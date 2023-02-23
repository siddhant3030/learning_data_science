import requests


url = f"https://kf.kobotoolbox.org/api/v2/assets.json"
response = requests.get(url, auth=("username", "password"))

json_response = response.json()

key_list=json_response.get('results')
a_key = "uid"

list_of_values = [a_dict[a_key] for a_dict in key_list]
print(list_of_values)
