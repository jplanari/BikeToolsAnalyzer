# src/data/strava.py
import requests
import io

def fetch_gpx_from_strava(activity_url, session_cookie):
    """
    Downloads GPX from Strava using a browser session cookie.
    """
    # 1. Clean the URL
    if "/export_gpx" not in activity_url:
        # Remove query parameters if any
        base_url = activity_url.split('?')[0]
        download_url = f"{base_url}/export_gpx"
    else:
        download_url = activity_url

    # 2. Set up headers with the cookie
    # Strava checks for a valid User-Agent and the specific session cookie
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": f"_strava4_session={session_cookie}"
    }

    try:
        response = requests.get(download_url, headers=headers)
        
        # 3. Check if we actually got the file or a redirect to login
        if response.status_code == 200:
            # If the content is HTML, it means the cookie failed (redirected to login)
            if b"<!DOCTYPE html>" in response.content[:100]:
                return None, "Cookie invalid or expired. Strava redirected to login."
            
            # Return a file-like object
            return io.BytesIO(response.content), None
        elif response.status_code == 404:
            return None, "Activity not found or you don't have permission to view it."
        else:
            return None, f"Error: {response.status_code}"
            
    except Exception as e:
        return None, str(e)
