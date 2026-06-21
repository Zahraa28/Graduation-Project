"""
carnet.py — CarNet free web endpoint.
Accepts raw image bytes for direct integration with uploaded files.
"""

import io
import requests
from PIL import Image
import pillow_heif

# Register HEIF for iPhone photo support
try:
    pillow_heif.register_heif_opener()
except ImportError:
    pass

def normalize_image(input_path) -> Image.Image:
    """Convert any format to a clean RGB PIL Object."""
    img = Image.open(input_path)
    
    # Handle transparency/Palette (PNG, WebP, etc.)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGBA")
        # make the background slightly gray
        canvas = Image.new("RGB", img.size, (128, 128, 128))
        canvas.paste(img, mask=img.split()[-1])
        return canvas
    
    return img.convert("RGB")


def resize_for_carnet(img: Image.Image, target_size=(1500, 844), max_kb=999) -> bytes:
    """
    Center Crop to 16:9 and scale to 1500x844. 
    This makes the car 'fill the frame' better than the padding method.
    """
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    img_aspect = img.width / img.height

    if img_aspect > target_aspect:
        # Image is wider than 16:9 - crop the sides
        new_width = int(target_aspect * img.height)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        # Image is taller than 16:9 - crop the top/bottom
        new_height = int(img.width / target_aspect)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))

    # Now resize to the exact target resolution
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Compression loop stays the same
    quality = 95
    while True:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", optimize=True, quality=quality)
        data = buffer.getvalue()
        size_kb = len(data) / 1024
        if size_kb <= max_kb or quality <= 15:
            print(f"Final Prep (Crop): {img.size[0]}x{img.size[1]} | {size_kb:.1f} KB")
            return data
        quality -= 5

# --- Prepare Image for upload ---
def get_ready_for_upload(image_file):
    # 1. Fix format
    clean_img = normalize_image(image_file)
    
    # 2. Fix size (< 1000 KB)
    final_jpeg_bytes = resize_for_carnet(clean_img)
    
    return final_jpeg_bytes

def call_carnet(file_path):
    """
    POST image bytes to CarNet endpoint.
    Returns dict: brand, model, generation, color, angle,
                  brand_prob, model_prob, generation_prob
    Returns {} on failure.
    """
    print(f"Processing: {file_path}...")
    try:
        # Step 1 & 2: Prepare the image (Resize to 1500x844 & < 1MB)
        img_ready = get_ready_for_upload(file_path)
        # DEBUG: Save the exact image being sent to see what the AI sees
        with open("debug_view.jpg", "wb") as f:
            f.write(img_ready)

        # Step 3: API Request
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://carnet.ai",
            "Referer": "https://carnet.ai/",
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Note: img_obj is already bytes from resize_for_carnet
        files = {"imageFile": ("car.jpg", img_ready, "image/jpeg")}
        
        resp = requests.post(
            "https://carnet.ai/recognize-file", 
            headers=headers, 
            files=files,
            timeout=15
        )
        
        if resp.status_code != 200:
            return {"error": f"Server error {resp.status_code}", "details": resp.text[:100]}

        data = resp.json()
        car = data.get("car")
        if not car: 
            return {"message": "No car found in image."}
        color_data = data.get("color", {})

        # --- Refined Confidence Logic ---
        # Fallback to general 'prob' if 'make_prob' is missing or 0
        raw   = float(car.get("make_prob", 0) or car.get("prob", 0))
        brand_prob = raw / 100
        model_prob = float(car.get("model_prob", raw)) / 100
        gen_prob   = float(car.get("gen_prob",   model_prob * 100)) / 100

        prediction = {
            "brand": car.get("make", ""),
            "model": car.get("model", ""),
            "year": car.get("generation", ""),
            "color": color_data.get("name") if isinstance(color_data, dict) else "—",
            "confidence": f"{float(raw):.1f}%",
            "brand_prob": round(brand_prob, 4),
            "model_prob": round(model_prob),
            "generation_prob": round(gen_prob, 4),
        }
        print(f"  CarNet → {prediction['brand']} {prediction['model']} ({prediction['brand_prob']:.0%})")

        return prediction
    except Exception as e:
        return {"error": str(e)}

# --- UPLOAD ONE IMAGE ---
#if __name__ == "__main__":
    # Change 'car_image.webp' to your actual file name
#    result = call_carnet("car3.webp")
#    print(result)