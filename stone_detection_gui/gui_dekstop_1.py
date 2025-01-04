import tkinter as tk
from tkinter import messagebox
import cv2
import threading
from PIL import Image, ImageTk

# Fungsi untuk memulai stream RTSP
def start_stream(ip, port):
    rtsp_url = f"rtsp://admin:@{ip}:{port}/stream"
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        messagebox.showerror("Error", "Tidak dapat terhubung ke RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame BGR ke RGB dan tampilkan pada Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        # Perbarui label gambar
        video_label.config(image=img_tk)
        video_label.image = img_tk

    cap.release()

# Fungsi untuk menangani input IP dan Port
def on_stream_button_click():
    ip = ip_entry.get()
    port = port_entry.get()

    if not ip or not port:
        messagebox.showerror("Error", "Harap masukkan IP dan Port")
        return

    # Jalankan stream dalam thread terpisah
    threading.Thread(target=start_stream, args=(ip, port), daemon=True).start()

# Fungsi untuk Preview
def on_preview_button_click():
    messagebox.showinfo("Preview", "ROI Area telah dipilih!")

# Fungsi untuk Analyze
def on_analyze_button_click():
    messagebox.showinfo("Analyze", "Model prediksi sedang dijalankan!")

# Create Tkinter window
window = tk.Tk()
window.title("CCTV Stream")
window.geometry("800x600")
window.resizable(False, False)

# Frame for input IP and Port
input_frame = tk.Frame(window)
input_frame.pack(pady=20)

# IP Address Label and Entry
tk.Label(input_frame, text="IP Address:").grid(row=0, column=0, padx=5, pady=5)
ip_entry = tk.Entry(input_frame)
ip_entry.grid(row=0, column=1, padx=5, pady=5)

# Port Label and Entry
tk.Label(input_frame, text="Port:").grid(row=0, column=2, padx=5, pady=5)
port_entry = tk.Entry(input_frame)
port_entry.grid(row=0, column=3, padx=5, pady=5)

# Stream Button
stream_button = tk.Button(input_frame, text="Stream", command=on_stream_button_click, width=10)
stream_button.grid(row=0, column=4, padx=5, pady=5)

# Label for video preview
video_label = tk.Label(window)
video_label.pack(pady=5)

# Label untuk preview video
video_label = tk.Label(window)
video_label.pack(pady=5)

# Frame untuk tombol Preview dan Analyze
action_frame = tk.Frame(window)
action_frame.pack(pady=20)

preview_button = tk.Button(action_frame, text="Preview", command=on_preview_button_click, width=10)
preview_button.grid(row=0, column=0, padx=10)

analyze_button = tk.Button(action_frame, text="Analyze", command=on_analyze_button_click, width=10)
analyze_button.grid(row=0, column=1, padx=10)

# Menjalankan aplikasi GUI
window.mainloop()