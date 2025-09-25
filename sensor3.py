
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import cnn2 as cnn

num_classes = 26
word_dict = {i: chr(65+i) for i in range(26)}

model_loaded = cnn.MappedCNN(
    (28, 28),
    [(3, 1, 1, 32), (3, 1, 1, 32), (3, 1, 1, 64), (3, 1, 1, 64)],
    [0, 2, 0, 2],
    [num_classes],
    1
)
model_loaded.load_state_dict(torch.load(
    "cnn_model2.pth", map_location="cpu"))


class Sensor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor")

        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.last_x, self.last_y = None, None
        self.drawing = False
        self.strokes = []

        self.pen_size = tk.IntVar(value=20)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # --- Controls ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(btn_frame, text="Clear", command=self.clear_canvas).pack(
            side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save", command=self.save_canvas).pack(
            side=tk.LEFT, padx=5)

        tk.Label(btn_frame, text="Pen size:").pack(side=tk.LEFT, padx=5)
        tk.Scale(btn_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                 variable=self.pen_size).pack(side=tk.LEFT)

        # Label for prediction result
        self.result_label = tk.Label(
            self.root, text="Prediction(Conductance): None", font=("Arial", 20))
        self.result_label.pack(pady=10)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing and self.last_x is not None and self.last_y is not None:
            size = self.pen_size.get()
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill="black", width=size, capstyle=tk.ROUND, smooth=True)
            self.strokes.append(
                (self.last_x, self.last_y, event.x, event.y, size))
            self.last_x, self.last_y = event.x, event.y

            # 🔑 Predict continuously while drawing
            self.predict_letter()

    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.strokes.clear()
        self.result_label.config(text="Prediction(Conductance): None")

    def predict_letter(self):
        if not self.strokes:
            return

        img = Image.new("L", (self.canvas.winfo_width(),
                        self.canvas.winfo_height()), "white")
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2, size in self.strokes:
            draw.line((x1, y1, x2, y2), fill="black", width=size)

        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array

        x = torch.tensor(img_array, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)

        with torch.no_grad():
            output = torch.argmax(model_loaded(x), dim=1)
            letter = word_dict[output.item()]

        self.result_label.config(text=f"Prediction(Conductance): {letter}")

    def save_canvas(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        output_width, output_height = 28, 28

        img = Image.new("RGB", (output_width, output_height), "white")
        draw = ImageDraw.Draw(img)

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale_x = output_width / canvas_w
        scale_y = output_height / canvas_h

        for x1, y1, x2, y2, size in self.strokes:
            draw.line(
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                fill="black", width=int(size * scale_x)  # scale pen size too
            )

        img.save(file_path, "PNG")

        print(f"Saved scaled canvas to {
              file_path} ({output_width}x{output_height})")


if __name__ == "__main__":
    root = tk.Tk()
    app = Sensor(root)
    root.mainloop()
