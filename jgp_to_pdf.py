# ------------------------------------------------------------
# Image to PDF Converter
# ------------------------------------------------------------
# This simple script allows you to:
# 1. Select multiple image files via a file dialog.
# 2. Choose where to save the output PDF.
# 3. Automatically combine all selected images into one PDF file.
# Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP
# ------------------------------------------------------------

from PIL import Image
from tkinter import Tk, filedialog

def convert_images_to_pdf():
    Tk().withdraw()

    files = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp")]
    )
    if not files:
        print("No images selected.")
        return

    save_path = filedialog.asksaveasfilename(
        title="Save PDF as",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not save_path:
        print("No save location selected.")
        return

    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:])
    print(f"PDF saved as: {save_path}")

if __name__ == "__main__":
    convert_images_to_pdf()
