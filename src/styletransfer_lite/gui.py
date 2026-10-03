"""Small desktop window. Nothing here runs until launch() is called."""
from __future__ import annotations

import threading
from typing import Optional

from .core import NUM_STYLES, StyleTransfer, StyleTransferError

PREVIEW = 420


def launch(device: str = "auto", image: Optional[str] = None) -> int:
    """Open the window. Returns 1 if no display or tkinter is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        from PIL import ImageTk
    except ImportError as exc:
        print(f"error: the GUI needs tkinter ({exc}). On Debian/Ubuntu: sudo apt install python3-tk")
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}). Use the command line instead.")
        return 1

    engine = StyleTransfer(device=device)
    root.title(f"styletransfer-lite ({engine.device})")
    state = {"photo": None, "style_path": None, "result": None, "tk": [None, None]}

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Button(top, text="Open photo...", command=lambda: open_photo()).pack(side="left")

    style_var = tk.StringVar(value="builtin")
    ttk.Radiobutton(top, text="Built-in style", variable=style_var, value="builtin").pack(side="left", padx=(12, 2))
    index_var = tk.IntVar(value=0)
    ttk.Spinbox(top, from_=0, to=NUM_STYLES - 1, width=4, textvariable=index_var).pack(side="left")
    ttk.Radiobutton(top, text="Style image", variable=style_var, value="image").pack(side="left", padx=(12, 2))
    style_btn = ttk.Button(top, text="Choose...", command=lambda: choose_style())
    style_btn.pack(side="left")

    row2 = ttk.Frame(root, padding=(8, 0))
    row2.pack(fill="x")
    ttk.Label(row2, text="Strength").pack(side="left")
    strength = tk.DoubleVar(value=1.0)
    ttk.Scale(row2, from_=0.0, to=1.0, variable=strength, length=160).pack(side="left", padx=6)
    keep = tk.BooleanVar(value=False)
    ttk.Checkbutton(row2, text="Keep original colours", variable=keep).pack(side="left", padx=8)
    go = ttk.Button(row2, text="Stylize", command=lambda: run())
    go.pack(side="left", padx=8)
    save_btn = ttk.Button(row2, text="Save result...", command=lambda: save(), state="disabled")
    save_btn.pack(side="left")

    panes = ttk.Frame(root, padding=8)
    panes.pack()
    left = ttk.Label(panes, text="Open a photo to start", width=48, anchor="center")
    right = ttk.Label(panes, text="", width=48, anchor="center")
    left.grid(row=0, column=0, padx=4)
    right.grid(row=0, column=1, padx=4)
    status = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=status, padding=(8, 0, 8, 8)).pack(fill="x")

    def show(label, img, slot):
        thumb = img.copy()
        thumb.thumbnail((PREVIEW, PREVIEW))
        state["tk"][slot] = ImageTk.PhotoImage(thumb)
        label.configure(image=state["tk"][slot], text="")

    def open_photo(path: Optional[str] = None):
        path = path or filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            from .core import load_image

            state["photo"] = path
            show(left, load_image(path), 0)
            status.set(f"Loaded {path}")
        except StyleTransferError as exc:
            messagebox.showerror("styletransfer-lite", str(exc))

    def choose_style():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            state["style_path"] = path
            style_var.set("image")
            style_btn.configure(text=path.replace("\\", "/").split("/")[-1][:24])

    def run():
        if not state["photo"]:
            messagebox.showinfo("styletransfer-lite", "Open a photo first.")
            return
        if style_var.get() == "image" and not state["style_path"]:
            messagebox.showinfo("styletransfer-lite", "Choose a style image first.")
            return
        go.configure(state="disabled")
        status.set("Working...")
        opts = dict(strength=round(strength.get(), 3), preserve_color=keep.get())
        photo, style_path, mode, idx = state["photo"], state["style_path"], style_var.get(), index_var.get()

        def work():
            try:
                if mode == "image":
                    result = engine.stylize_with(photo, style_path, **opts)
                else:
                    result = engine.stylize(photo, style=idx, **opts)
                root.after(0, lambda: done(result, None))
            except Exception as exc:  # shown to the user, not swallowed
                root.after(0, lambda: done(None, exc))

        threading.Thread(target=work, daemon=True).start()

    def done(result, error):
        go.configure(state="normal")
        if error is not None:
            status.set("Failed.")
            messagebox.showerror("styletransfer-lite", str(error))
            return
        state["result"] = result
        show(right, result, 1)
        save_btn.configure(state="normal")
        status.set(f"Done ({result.size[0]}x{result.size[1]}).")

    def save():
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path and state["result"] is not None:
            state["result"].save(path)
            status.set(f"Saved {path}")

    if image:
        root.after(100, lambda: open_photo(image))
    root.mainloop()
    return 0
