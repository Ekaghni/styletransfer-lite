"""Capture the GUI for the README: python tools/capture_gui.py assets/gui.png"""
import ctypes
import sys
import tkinter as tk

from PIL import ImageGrab

ctypes.windll.shcore.SetProcessDpiAwareness(2)
out = sys.argv[1]
orig_mainloop = tk.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def fake_mainloop(self, n=0):
    def stylize():
        for w in walk(self):
            if w.winfo_class() == "TButton" and w.cget("text") == "Stylize":
                w.invoke()
    def spin():
        for w in walk(self):
            if w.winfo_class() == "TSpinbox":
                w.set(19)
    def grab():
        self.update()
        x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        ImageGrab.grab((x, y, x + w, y + h)).save(out)
        self.destroy()
    self.lift(); self.attributes("-topmost", True)
    self.after(600, spin)
    self.after(900, stylize)
    self.after(3500, grab)
    orig_mainloop(self, n)


tk.Tk.mainloop = fake_mainloop
from styletransfer_lite.gui import launch
launch(device="cpu", image="examples/astronaut.jpg")
