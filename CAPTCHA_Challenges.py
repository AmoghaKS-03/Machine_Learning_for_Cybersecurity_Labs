import random
import string
import os
import pytesseract

from captcha.image import ImageCaptcha
from PIL import Image, ImageFilter

# EXPLICITLY TELL PYTESSERACT WHERE THE EXECUTABLE IS:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 1. Create CAPTCHA image
os.makedirs("captchas", exist_ok=True)

characters = string.ascii_uppercase + string.digits
captcha_text = "".join(
    random.choices(characters, k=5)
)

generator = ImageCaptcha(width=200, height=80)

image_path = "captchas/sample.png"
generator.write(captcha_text, image_path)

print("Actual CAPTCHA:", captcha_text)

# 2. Load and preprocess image
image = Image.open(image_path).convert("L")
image = image.filter(ImageFilter.SHARPEN)
image = image.point(
    lambda pixel: 255 if pixel > 140 else 0
)

# 3. Recognize CAPTCHA using OCR
predicted_text = pytesseract.image_to_string(
    image,
    config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
).strip().replace(" ", "").replace("\n", "")

print("Predicted CAPTCHA:", predicted_text)

# 4. Evaluate prediction
if predicted_text == captcha_text:
    print("Result: CAPTCHA recognized correctly")
else:
    print("Result: CAPTCHA recognition failed")

# 5. Display CAPTCHA
display_image = Image.open(image_path)
display_image.show()