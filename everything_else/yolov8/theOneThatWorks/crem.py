import base64

encoded_str = "AQzYzZz0VNbhXRXR2MFtFa8FFcl13O0glT8kVZ6R1WRRTU1gmSVRDV=="

try:
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_str = decoded_bytes.decode('utf-8')
    print(f"Base64 Decoded String: {decoded_str}")
except Exception as e:
    print(f"Error decoding Base64: {e}")
