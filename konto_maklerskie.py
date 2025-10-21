import requests

XTB_API_URL = "https://xapi.xtb.com"
LOGIN = "51502770"
PASSWORD = "Mati86mati"

def login():
    url = f"{XTB_API_URL}/api/v2/login"  # Poprawny endpoint
    payload = {
        "userId": LOGIN,
        "password": PASSWORD
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10, verify=False)  # Wyłącz SSL dla testów
        response.raise_for_status()
        data = response.json()
        if data.get("status"):
            return data["streamSessionId"]
        else:
            print("Błąd logowania:", data)
            return None
    except requests.exceptions.RequestException as e:
        print("Błąd połączenia:", e)
        return None

session_id = login()
if session_id:
    print("✅ Zalogowano! ID sesji:", session_id)
else:
    print("❌ Nie udało się zalogować.")
