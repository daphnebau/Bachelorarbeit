
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from numba import njit, prange

# This code generates a Mandelbrot Set and the Julia set for z^2+c. At first, by default c=0. 
# By moving the cursor on the left hand side of the screen, one can vary c and see the corresponding Julia set on the right. 
# One may notice that for parameters inside the Mandelbrot set, the corresponding Julia set is connected, and outside it becomes disconnected.
# By theorem 4.4.1., one can see that the Julia sets move holomorphically everywhere except for on the boundary of the Mandelbrot set.


#this function checks if a parameter c belongs to M
@njit
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2: #because of Lemma 3.7.2. this determines whether c belongs to M
            return n
        z = z * z + c
    return max_iter

#this function checks for a full grid of parameters c if they lie in M
@njit(parallel=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width) # coordinate grid of the region of the plane in which we want to check if parameter c lies in Mandelbrot set
    r2 = np.linspace(ymin, ymax, height) 
    img = np.empty((height, width), dtype=np.uint32) #2D array that will hold the pixel values of the Mandelbrot set image
    for i in prange(height):
        for j in prange(width):
            c = complex(r1[j], r2[i])
            img[i, j] = mandelbrot(c, max_iter) #determines whether or not the point belongs to the Mandelbrot set
    return (r1, r2, img)


# Function to determine if a point is on the boundary of Julia set
@njit
def julia_boundary(c, z, max_iter):
    for n in range(max_iter):
        if abs(z) > 2:
            return n #returns how fast one can determine that the iterated function becomes unbounded
        z = z * z + c
    return -1  # Return -1 if it doesn't escape

# Function that checks for a full grid of parameters c if the lie in Julia set
@njit(parallel=True)
def julia_set_boundary(c, xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width) # coordinate grid 
    r2 = np.linspace(ymin, ymax, height)
    img = np.empty((height, width), dtype=np.uint32)
    for i in prange(height):
        for j in prange(width):
            z = complex(r1[j], r2[i])
            result = julia_boundary(c, z, max_iter)   #check if z is on the boundary of J(x^2+c)
            if result >= 0:
        
                img[i, j] = result #result gives us how fast it diverges to infinity
            else:
                img[i, j] = 0  # Use 0 for points that do not escape
    return (r1, r2, img)


#definition of the App showing Mandelbrot and Julia set
class MandelbrotJuliaApp:
    def __init__(self, root):
        self.root = root #main window of the app
        self.root.title("Mandelbrot and Julia Set Viewer")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5)) #creates a figure with 2 side by side plots

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) #area where the Julia set is rendered

        self.max_iter = 1000 #max number of iterations for Julia set
        self.max_iter_mandel = 5000 #max number of iterations for Mandelbrot set
        self.width, self.height = 800, 800 #resolution of plot
        # Center the Mandelbrot set around 0
        self.xmin, self.xmax = -2.5, 1.5
        self.ymin, self.ymax = -2.0, 2.0
        self.c = complex(0, 0) #initialises c as 0
    

        self.update_mandelbrot()
        self.update_julia()

        self.canvas.draw()

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick) #connects a mouse click event to the onclick method 
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion) # connects mouse movement to the onmotion method

    #generates and displays the Mandelbrot set on the left plot
    def update_mandelbrot(self):
        r1, r2, img = mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter_mandel)
        self.ax1.clear()
        self.ax1.imshow(img, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap='magma')
        
        self.ax1.set_title("Mandelbrot Set")

    #generates and displays the Julia set corresponding to the current complex parameter c on the right plot
    def update_julia(self):
        # Center the Julia set around the current c
        c_real = self.c.real
        c_imag = self.c.imag
        # Adjust the plot range
        center_x, center_y = 0, 0
        extent = 2.0  # You can adjust this to zoom in or out
        
        r1, r2, img = julia_set_boundary(self.c, center_x - extent, center_x + extent, center_y - extent, center_y + extent, self.width, self.height, self.max_iter)
        
        self.ax2.clear()
        self.ax2.imshow(img, extent=(center_x - extent, center_x + extent, center_y - extent, center_y + extent), cmap='magma')
        self.ax2.set_title(f"Julia Set for c = {self.c}")

    #This method handles mouse click events, updates Julia set on click of a paramter c 
    def onclick(self, event):
        if event.inaxes == self.ax1:
            x = event.xdata
            y = event.ydata
            real = x
            imag = y
            self.c = complex(real, imag)
            self.update_julia()
            self.canvas.draw()

    #dynamically updates the Julia set
    def onmotion(self, event):
        if event.inaxes == self.ax1:
            if event.xdata is not None and event.ydata is not None:
                # Update c based on cursor position
                real = event.xdata
                imag = event.ydata
                self.c = complex(real, imag)
                self.update_julia()
                self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk() #initialises application and sets main window
    app = MandelbrotJuliaApp(root) #creates an instance of the MandelbrotJuliaApp class
    root.mainloop() #mainloop() method enters an infinite loop that waits for events (like clicking on parameter c) and handles them as they occur
