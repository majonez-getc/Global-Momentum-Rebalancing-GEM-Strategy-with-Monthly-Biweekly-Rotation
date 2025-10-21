from PIL import Image
import os

def convert_images_to_pdf(image_paths, output_pdf_path):
    images = []

    for img_path in image_paths:
        img = Image.open(img_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        images.append(img)

    if images:
        images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
        print(f"PDF zapisany jako: {output_pdf_path}")
    else:
        print("Brak poprawnych obrazów.")

if __name__ == "__main__":
    # Przykład: wrzuć obrazy do folderu 'obrazy'
    folder = r'C:\Users\agent\Downloads\Telegram Desktop\zdj'
    output_pdf = "wynik.pdf"

    image_files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    convert_images_to_pdf(image_files, output_pdf)
