import tkinter as tk
from tkinter import filedialog, image_names
from tkinter.constants import BOTTOM, LEFT, NW, RIGHT
from PIL import ImageTk,Image
from skimage import data, feature, filters

class Application(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()        

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width = 800, height = 800)      
        self.canvas.pack(side=BOTTOM)

        camera = data.camera()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(camera))

        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Abrir Imagem ↑"
        self.hi_there["command"] = self.open_image
        self.hi_there["fg"] = "blue"
        self.hi_there.pack(side=LEFT, padx=10, pady=5)

        self.hi_there2 = tk.Button(self)
        self.hi_there2["text"] = "Salvar imagem ↓"
        self.hi_there2["command"] = self.save_image
        self.hi_there2.pack(side=LEFT, padx=10, pady=5)
        
        self.hi_there3 = tk.Button(self)
        self.hi_there3["text"] = "3Hello World\n(click me)"
        self.hi_there3["command"] = self.say_hi
        self.hi_there3.pack(side=LEFT, padx=10, pady=5)

        self.hi_there4 = tk.Button(self)
        self.hi_there4["text"] = "4Hello World\n(click me)"
        self.hi_there4["command"] = self.say_hi
        self.hi_there4.pack(side=LEFT, padx=10, pady=5)

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side=RIGHT, padx=100, pady=5)

    def say_hi(self):
        print("hi there, everyone!")

    def open_image(self):
        self.file = filedialog.askopenfilename(initialdir = "/Imagem", filetypes=[('image files', ('.png', '.jpg', '.webp'))])
        self.img = ImageTk.PhotoImage(Image.open(self.file))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def save_image(self):
        imgpil = ImageTk.getimage(self.img)
        directory = filedialog.asksaveasfilename(initialfile="Sem_Título",title='Salvar imagem', filetypes=[('PNG image', '.png'), ('JPG image', '.jpg'), ('WEBP image', '.webp')])
        if not directory:
            return

        if not (directory.endswith('.png') or directory.endswith('.jpg') or directory.endswith('.webp')):
            directory += '.png'
        imgpil.save(directory, format="png")

root = tk.Tk()
app = Application(master=root)
app.master.title("My Do-Nothing Application")
app.master.minsize(1200, 900)
app.master.maxsize(1200, 900)
app.mainloop()