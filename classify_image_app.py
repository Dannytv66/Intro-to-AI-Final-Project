import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import tensorflow as tf

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------------
# Load the saved model
# -------------------------------
MODEL_PATH = "cifar10_cnn_3block.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Modern Dark Theme Colors
# -------------------------------
BG = "#1E1E1E"
FG = "#FFFFFF"
ACCENT = "#3A86FF"
CARD = "#2C2C2C"

# -------------------------------
# Main Window
# -------------------------------
root = tk.Tk()
root.title("CIFAR-10 Image Classifier")
root.geometry("700x820")
root.configure(bg=BG)

# Nice font settings
header_font = ("Segoe UI", 20, "bold")
button_font = ("Segoe UI", 14)
text_font = ("Consolas", 12)

# -------------------------------
# Title
# -------------------------------
title_label = tk.Label(
    root, text="CIFAR-10 Image Classifier",
    bg=BG, fg=FG, font=header_font
)
title_label.pack(pady=20)

# -------------------------------
# Image Display Card
# -------------------------------
image_frame = tk.Frame(root, bg=CARD, bd=2, relief="ridge")
image_frame.pack(pady=15)

image_label = tk.Label(image_frame, bg=CARD)
image_label.pack(padx=10, pady=10)

# -------------------------------
# Prediction Text Box
# -------------------------------
prediction_frame = tk.Frame(root, bg=CARD)
prediction_frame.pack(pady=15, fill="x", padx=20)

prediction_text = tk.Label(
    prediction_frame,
    text="Upload or paste an image to classify it.",
    bg=CARD, fg=FG, font=text_font, justify="left"
)
prediction_text.pack(padx=10, pady=10)

# -------------------------------
# Helper: preprocess image
# -------------------------------
def preprocess_image(img):
    img = img.resize((32, 32))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Prediction Logic
# -------------------------------
def classify_image(img):
    processed = preprocess_image(img)
    preds = model.predict(processed)[0]

    top_class = np.argmax(preds)
    confidence = preds[top_class] * 100

    text = f"Prediction → {CLASS_NAMES[top_class].upper()}  ({confidence:.2f}% confidence)\n\n"
    text += "Class Probabilities:\n"

    for i, p in enumerate(preds):
        text += f"{CLASS_NAMES[i]:<12}: {p*100:.2f}%\n"

    prediction_text.config(text=text)

# -------------------------------
# Display + classify
# -------------------------------
def show_and_classify(img):
    preview = img.resize((300, 300))
    imgtk = ImageTk.PhotoImage(preview)
    image_label.config(image=imgtk)
    image_label.image = imgtk

    classify_image(img)

# -------------------------------
# Load from File
# -------------------------------
def load_image_file():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )
    if not path:
        return
    img = Image.open(path).convert("RGB")
    show_and_classify(img)

# -------------------------------
# Load from Clipboard
# -------------------------------
def load_from_clipboard():
    try:
        img = ImageGrab.grabclipboard()
        if img is None:
            messagebox.showerror("Error", "No image found in clipboard.")
            return
        img = img.convert("RGB")
        show_and_classify(img)
    except Exception as e:
        messagebox.showerror("Error", f"Clipboard read failed:\n{e}")

# -------------------------------
# Modern Rounded Buttons
# -------------------------------
def modern_button(parent, text, command):
    return tk.Button(
        parent, text=text, command=command,
        bg=ACCENT, fg="white",
        activebackground="#1B5ED2",
        activeforeground="white",
        font=button_font,
        bd=0, padx=20, pady=10,
        relief="flat",
        highlightthickness=0
    )

btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=10)

upload_btn = modern_button(btn_frame, "Upload Image", load_image_file)
upload_btn.grid(row=0, column=0, padx=10)

paste_btn = modern_button(btn_frame, "Paste From Clipboard", load_from_clipboard)
paste_btn.grid(row=0, column=1, padx=10)

# -------------------------------
# Run GUI
# -------------------------------
root.mainloop()
