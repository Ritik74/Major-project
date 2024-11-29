import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from keras import layers, models
from PIL import Image, ImageDraw

# Load the MNIST dataset from Keras
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
train_data = train_data / 255.0
test_data = test_data / 255.0

# Reshape data to include the channel dimension for CNN (28x28x1)
train_data = train_data.reshape((-1, 28, 28, 1))
test_data = test_data.reshape((-1, 28, 28, 1))

# Create a Convolutional Neural Network (CNN) model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output 10 classes (for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=100)

# Create the tkinter window for drawing
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognition - ML Project")

        # Add heading above the canvas
        self.heading_label = tk.Label(self.master, text="ML-Based Major Project: Digit Recognition\n"
                                                        "Accuracy: ~95% (Not 100% Accurate)",
                                      font=("Helvetica", 14), fg="blue")
        self.heading_label.grid(row=0, column=0, padx=10, pady=10)

        # Canvas for drawing
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="white")
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        # Clear button
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=10)

        # Recognize button
        self.recognize_button = tk.Button(self.master, text="Recognize", command=self.recognize_digit)
        self.recognize_button.grid(row=3, column=0, pady=10)

        # Variables to track drawing state
        self.has_drawn = False
        self.drawing_active = False

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.check_lift)

    def draw(self, event):
        if not self.has_drawn:
            self.has_drawn = True  # User starts drawing

        if self.drawing_active:
            messagebox.showwarning(
                "Drawing Alert",
                "You have already drawn something. Clear the canvas before drawing again."
            )
            return

        x, y = event.x, event.y
        r = 10  # Radius of the circle
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", width=3)

    def check_lift(self, event):
        # Mark that the user has finished their first stroke
        if self.has_drawn:
            self.drawing_active = True

    def clear_canvas(self):
        self.canvas.delete("all")
        self.has_drawn = False
        self.drawing_active = False

    def get_image(self):
        # Create a new image from the Tkinter canvas
        img = Image.new("L", (280, 280), 255)  # Create a white 280x280 image
        draw = ImageDraw.Draw(img)

        # Get all the coordinates from the canvas
        for item in self.canvas.find_all():
            x1, y1, x2, y2 = self.canvas.coords(item)
            draw.line([x1, y1, x2, y2], fill="black", width=3)

        # Resize image to 28x28
        img = img.resize((28, 28))

        # Convert to a NumPy array
        img = np.array(img)

        # Invert the image (canvas is white with black lines, we want black on white)
        img = np.invert(img)

        # Normalize the image
        img = img / 255.0

        # Reshape the image to include the channel dimension (for CNN)
        img = img.reshape(1, 28, 28, 1)

        return img, img

    def recognize_digit(self):
        img, original_img = self.get_image()

        # Check if the image has enough non-white pixels (simple check for drawing)
        non_white_pixels = np.sum(original_img < 128)  # Count non-white pixels (black pixels)

        # If too few non-white pixels, the drawing is probably invalid or empty
        if non_white_pixels < 500:  # Adjust threshold if needed
            messagebox.showwarning("Invalid Input", "Please draw a valid digit (0-9).")
            return

        # Predict the drawn digit
        prediction_probabilities = model.predict(img)
        predicted_class = np.argmax(prediction_probabilities, axis=1)[0]
        prediction_confidence = np.max(prediction_probabilities)

        # Check if the model's confidence is low (e.g., below a threshold)
        if prediction_confidence < 0.7:  # Adjust threshold as needed
            messagebox.showwarning(
                "Invalid Input", "Please draw a valid digit (0-9)."
            )
            return

        # Display the result with accuracy as percentage
        accuracy = prediction_confidence * 100
        messagebox.showinfo("Prediction", f"Predicted Digit: {predicted_class}\nAccuracy: {accuracy:.2f}%")

# Run the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()     