import requests
import os

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fetch_data():
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,T2M,T2MDEW,T2MWET,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,QV2M,WS2M,GWETTOP",
        "community": "AG",
        "longitude": "120.9751",
        "latitude": "14.5822",
        "start": "19950101",
        "end": "20251231",
        "format": "CSV"
    }
    
    target_path = os.path.join(BASE_DIR, "data", "raw", "Manila.csv")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    print(f"Fetching data to {target_path}...")
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        print("Success: Data fetched.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    fetch_data()
