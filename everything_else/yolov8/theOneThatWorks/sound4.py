import cv2
import numpy as np
import urllib.request

def fetch_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

def compute_depth_map(left_img, right_img, baseline, focal_length):
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5,  # Adjust
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity map
    disparity = stereo.compute(left_gray, right_gray)

    # Normalize for display
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Calculate depth map (basic formula)
    depth_map = np.zeros_like(disparity, np.float32)
    depth_map[disparity > 0] = (baseline * focal_length) / disparity[disparity > 0]

    return depth_map, disparity_normalized

# URLs of the stereo images
url1 = 'http://192.168.0.207/640x480.jpg'  # Left Camera
url2 = 'http://192.168.0.154/640x480.jpg'  # Right Camera

# Fetch images
left_image = fetch_image_from_url(url1)
right_image = fetch_image_from_url(url2)

# Stereo camera parameters (example values, you need actual calibrated values)
baseline = 0.06  # Distance between the two cameras [meters]
focal_length = 700  # Focal length of the camera [pixels]

# Compute depth and disparity maps
depth_map, disparity_map = compute_depth_map(left_image, right_image, baseline, focal_length)

# Display results
cv2.imshow("Left Image", left_image)
cv2.imshow("Right Image", right_image)
cv2.imshow("Disparity Map", disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()