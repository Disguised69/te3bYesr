import base64
from PIL import Image

def convert_img_to_base64(imageNumpy):
    # Assuming img_array is your numpy array image
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(imageNumpy)

    # Convert PIL Image to bytes
    img_bytes = img_pil.tobytes()

    # Encode bytes to base64
    base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    return base64_encoded
