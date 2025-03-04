import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [63,1,145,233,1,0,150,0,2.3,0,0,1,1]} # Presence
data = {"features": [50,0,130,200,0,1,150,1,1.5,0,1,0,0]} # Absence

response = requests.post(url, json=data)
print(response.json())
