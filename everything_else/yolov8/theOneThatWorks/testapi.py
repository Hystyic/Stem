import requests

def query_google_lens(api_key, image_path):
    """
    Query Google Lens via SERPAPI and return the response.
    """
    print(f"Querying Google Lens with {image_path}")
    endpoint = 'https://serpapi.com/search?engine=google_lens'
    params = {
        'engine': 'google_lens',
        'api_key': api_key
    }
    files = {'image': open(image_path, 'rb')}
    response = requests.post(endpoint, files=files, params=params)
    return response

# Replace with your SERPAPI key and the path to your test image
api_key = '7365173579649d9ebd3d4aae9e00a879703ea43d1c8787a2b8f92244206d329f'  # Replace with your actual API key
test_image_path = 'https://www.ikea.com/in/en/images/products/hemlagad-pot-with-lid-black__0789061_pe763799_s5.jpg'  # Replace with the path to your test image

# Query Google Lens and print the response
response = query_google_lens(api_key, test_image_path)
if response.status_code == 200:
    print("API Query Successful. Response:")
    print(response.json())
else:
    print("Error in querying Google Lens.")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
