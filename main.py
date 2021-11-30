from numpy.core.records import array
from scipy.ndimage.filters import gaussian_filter
from skimage.restoration import non_local_means
from skimage.util import dtype
from funcs import *
from multidim_idct import *
import tkinter as tk
from tkinter import filedialog, image_names
from tkinter.constants import BOTTOM, LEFT, NW, RIGHT, TOP
from tkinter.simpledialog import askinteger
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import ImageTk, Image
from skimage import data, feature, filters
import numpy as np

class Application(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()  
        self.createMenu()      
    
    def createMenu(self):
        self.menubar = tk.Menu(self.master)

        self.createFileMenu()
        self.createEffectsMenu()
        self.createDefectMenu()
        self.createImageMenu()
        self.createTransformMenu()

        self.master.config(menu = self.menubar)

    def createFileMenu(self):
        file = tk.Menu(self.menubar, tearoff = 0)
        self.menubar.add_cascade(label ='Arquivo', menu = file)
        file.add_command(label ='Abrir imagem...', command = self.open_image)
        file.add_command(label ='Abrir câmera', command = self.open_camera)
        file.add_command(label ='Abrir moedas', command = self.open_coins)
        file.add_command(label ='Salvar como...', command = self.save_image)
        file.add_command(label ='Salvar rápido como JPG', command = self.save_image_jpg)
        file.add_separator()
        file.add_command(label ='Sair', command = self.master.destroy, foreground='red')

    def createEffectsMenu(self):
        menu = tk.Menu(self.menubar, tearoff = 0)
        self.menubar.add_cascade(label ='Efeitos', menu = menu)
        menu.add_command(label ='Blur...', command =  self.call_simple_blur)
        menu.add_command(label ='Mosaic...', command =  self.call_mosaic)
        menu.add_command(label ='Match color histogram...', command =  self.call_match_operation)
        menu.add_command(label ='Gaussian Blur', command =  self.call_gaussian_blur)
        menu.add_command(label ='Unsharp Mask (estranho)', command =  self.call_unsharp_mask)
        menu.add_command(label ='Median Filter', command =  self.call_median_filter)
        menu.add_command(label ='Non Local Means', command =  self.call_non_local_means)

        menuSegBinaria = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label ="Segmentação Binária", menu = menuSegBinaria)
        menuSegBinaria.add_command(label ="Isodata", command= self.call_segBin_Isodata)
        menuSegBinaria.add_command(label ="Li", command= self.call_segBin_Li)
        menuSegBinaria.add_command(label ="Mean", command= self.call_segBin_Mean)
        menuSegBinaria.add_command(label ="Minimum", command= self.call_segBin_Minimum)
        menuSegBinaria.add_command(label ="Otsu", command= self.call_segBin_Otsu)
        menuSegBinaria.add_command(label ="Triangulo", command= self.call_segBin_Triangle)
        menuSegBinaria.add_command(label ="Yen", command= self.call_segBin_Yen)
        menuSegBinaria.add_command(label ="Customizado...", command= self.call_segBin_Custon)

        menuDetecBordas = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label = "Detecção de Bordas", menu = menuDetecBordas)
        menuDetecBordas.add_command(label ="Usando Kernel", command= self.call_detecBordas_Kernel)
        menuDetecBordas.add_command(label ="Roberts", command= lambda: self.call_detecBordas(detecRoberts))
        menuDetecBordas.add_command(label ="Sobel", command= lambda: self.call_detecBordas(detecSobel))
        menuDetecBordas.add_command(label ="Scharr", command= lambda: self.call_detecBordas(detecScharr))
        menuDetecBordas.add_command(label ="Prewitt", command= lambda: self.call_detecBordas(detecPrewitt))
        menuDetecBordas.add_command(label ="Farid", command= lambda: self.call_detecBordas(detecFarid))
        menuDetecBordas.add_command(label ="Canny", command= lambda: self.call_detecBordas(detecCanny))

        menuDetection = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Detecção", menu=menuDetection)
        menuDetection.add_command(label="Linhas", command=self.call_line_detection)
        menuDetection.add_command(label="Linhas Probabilística", command=self.call_probabilistic_line_detection)
        menuDetection.add_command(label="Circulos", command=self.call_circle_detection)
        menuDetection.add_command(label="Componentes", command=lambda: self.call_component_detection(False))
        menuDetection.add_command(label="Propriedades", command=lambda: self.call_component_detection(True))

    def createDefectMenu(self):
        menu = tk.Menu(self.menubar, tearoff= 0)
        self.menubar.add_cascade(label ='Defeitos', menu = menu)
        menu.add_command(label ='Sal e Pimenta', command =  self.call_sp)
        menu.add_command(label ='Ruído', command =  self.call_noise)

    def createImageMenu(self):
        menu = tk.Menu(self.menubar, tearoff = 0)
        self.menubar.add_cascade(label ='Imagem', menu = menu)
        menu.add_command(label ='Rotacionar...', command =  self.call_rotation)
        menu.add_command(label ='Histograma...', command = self.draw_3hist)
        menu.add_command(label ='Espelhar Horizontalmente', command = self.call_flip_hor)
        menu.add_command(label ='Espelhar Verticalmente', command = self.call_flip_ver)
        menu.add_command(label="Quantizar", command=self.quantizarImg)

    def createTransformMenu(self):
        menu = tk.Menu(self.menubar, tearoff = 0)
        self.menubar.add_cascade(label ='Fourier', menu = menu)
        menuTransformada = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label ="Transformadas", menu = menuTransformada)

        menuTransformada.add_command(label="Fourier Passa alta", command = lambda: self.aplicaFFT("alta"))
        menuTransformada.add_command(label="Fourier Passa baixa", command = lambda: self.aplicaFFT("baixa"))

    def aplicaFFT(self, tipo):
        im = np.array(ImageTk.getimage(self.img))
        im = im[:,:,0]
        ft, mascara, dft = fourrierTransform(im, tipo)
        # print(mascara)
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        ax1.imshow(ft, cmap="gray")
        ax1.set_title('Transformada de Fourier')
        ax2.imshow(mascara[:,:,1], cmap="gray")
        ax2.set_title('Mascara passa ' + tipo)
        ax3.imshow(dft, cmap="gray")
        ax3.set_title('Img')
        plt.show()

    def quantizarImg(self):
        qtdQuant = tk.simpledialog.askinteger("Input", "Insira quanto ira quantizar", parent=self.master, minvalue=0)
        if qtdQuant == None:
            return
        im = np.array(ImageTk.getimage(self.img))
        imq = encode_quant(im, qtdQuant)
        im = decode_quant(imq,qtdQuant)
        self.img = im
        self.desenhar_imagemRGB(self.img)

    def desenhar_imagemRGB(self, imagem):
        self.img = ImageTk.PhotoImage(image=Image.fromarray(imagem.astype('uint8'), 'RGBA'))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def desenhar_imagem_grayscale(self, imagem):
        self.img = ImageTk.PhotoImage(image=Image.fromarray(imagem.astype('uint8'), 'L'))

    def desenhar_imagemRGBFloat(self, imagem):
        print(imagem)
        formatted = (imagem * 255 / np.max(imagem)).astype('uint8')
        formatted[:, :, 1] = formatted[:, :, 0]
        formatted[:, :, 2] = formatted[:, :, 0]
        self.img = ImageTk.PhotoImage(Image.fromarray(formatted))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width = 800, height = 800)      
        self.canvas.pack(side=BOTTOM)

        camera = data.camera()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(camera))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def open_image(self):
        self.file = filedialog.askopenfilename(initialdir = "/Imagem", filetypes=[('image files', ('.png', '.jpg', '.webp'))])
        if self.file == "":
            return
        self.img = ImageTk.PhotoImage(Image.open(self.file))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def open_camera(self):
        camera = data.camera()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(camera))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def open_coins(self):
        moedas = data.coins()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(moedas))
        self.canvas.create_image(20, 20, anchor=NW, image=self.img)

    def save_image_jpg(self):
        im = np.array(ImageTk.getimage(self.img))
        im = im[:,:, :3]

        x = np.zeros(im.shape, dtype=int)
        x[:, :, 0] = im[:, :, 2]
        x[:, :, 1] = im[:, :, 1]
        x[:, :, 2] = im[:, :, 0]

        quants = [1] #[0.5, 1, 2, 5, 10]
        blocks = [(8,8)] #[(2, 8), (8, 8), (16, 16), (32, 32), (200, 200)]
        for qscale in quants:
            for bx, by in blocks:
                quant = (
                    (np.ones((bx, by)) * (qscale * qscale))
                    .clip(-100, 100)  # to prevent clipping
                    .reshape((1, bx, 1, by, 1))
                )
                enc = encode_dct(x, bx, by)
                encq = encode_quant(enc, quant)
                encz = encode_zip(encq)
                decz = decode_zip(encz, encq.shape)
                decq = decode_quant(encq, quant)
                dec = decode_dct(decq, bx, by)
                cv2.imwrite("IMG.png", dec.astype(np.uint8))                
        
    def save_image(self):
        imgpil = ImageTk.getimage(self.img)
        directory = filedialog.asksaveasfilename(initialfile="Sem_Título",title='Salvar imagem', filetypes=[('PNG image', '.png'), ('JPG image', '.jpg'), ('WEBP image', '.webp')])
        if not directory:
            return

        if not (directory.endswith('.png') or directory.endswith('.jpg') or directory.endswith('.webp')):
            directory += '.png'
        imgpil.save(directory, format="png")

    def call_simple_blur(self):
        nivel_blur = tk.simpledialog.askinteger("Input", "Insira o nível do blur", parent=self.master, minvalue=0, maxvalue=20)
        if nivel_blur == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = simple_blur(array_imagem, nivel_blur)
        # simple_blur(array_imagem, nivel_blur)
        # print(nova_imagem[0][0])
        self.desenhar_imagemRGB(nova_imagem)

    def call_mosaic(self):
        nivel_blur = tk.simpledialog.askinteger("Input", "Insira o nível do mosaico", parent=self.master, minvalue=0, maxvalue=20)
        if nivel_blur == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = mosaic(array_imagem, nivel_blur)
        # simple_blur(array_imagem, nivel_blur)
        # print(nova_imagem[0][0])
        self.desenhar_imagemRGB(nova_imagem)

    def call_rotation(self):
        angulo = tk.simpledialog.askinteger("Input", "Insira o valor da rotação em graus", parent=self.master, minvalue=-360, maxvalue=360)
        if angulo == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = rotate_img(array_imagem, angulo)
        self.desenhar_imagemRGB(nova_imagem)

    def draw_3hist(self):
        img = np.array(ImageTk.getimage(self.img))
        plt.hist(img[:,:,0].ravel(), 256, [0,256], color='red', alpha=0.5)
        plt.hist(img[:,:,1].ravel(), 256, [0,256], color='green', alpha=0.5)
        plt.hist(img[:,:,2].ravel(), 256, [0,256], color='blue', alpha=0.5)
        plt.show()

    
    def call_match_operation(self):
        array_imagem1 = np.array(ImageTk.getimage(self.img))
        file = filedialog.askopenfilename(initialdir = "/Imagem", filetypes=[('image files', ('.png', '.jpg', '.webp'))])
        # print(type(self.img))
        if file == "":
            return
        imagem2 = ImageTk.PhotoImage(Image.open(file))
        # print(type(imagem2))
        array_imagem2 = np.array(ImageTk.getimage(imagem2))
        # print(array_imagem1.shape)
        # print(array_imagem2.shape)
        nova_imagem = match_operation(array_imagem1, array_imagem2)
        self.desenhar_imagemRGB(nova_imagem)

    def call_flip_hor(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = flip_hor(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_flip_ver(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = flip_ver(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_filtro_gaussiano(self):
        nivel_blur = tk.simpledialog.askinteger("Input", "Insira o sigma", parent=self.master, minvalue=0, maxvalue=40)
        if nivel_blur == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = filtroGaussiano(array_imagem, nivel_blur)
        self.desenhar_imagemRGB(nova_imagem)

    def call_gaussian_blur(self):
        nivel_blur = tk.simpledialog.askinteger("Input", "Insira o sigma", parent=self.master, minvalue=0, maxvalue=40)
        if nivel_blur == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = gaussian_filter_default(array_imagem, nivel_blur)
        self.desenhar_imagemRGB(nova_imagem)

    def call_sp(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = salt_and_peper(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_noise(self):
        sigma = tk.simpledialog.askinteger("Input", "Insira o sigma [0 - 100]", parent=self.master, minvalue=0, maxvalue=100)
        if sigma == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = noise(array_imagem, sigma)
        self.desenhar_imagemRGB(nova_imagem)

    def call_unsharp_mask(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = unsharp_mask2(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_median_filter(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = median_filter(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_non_local_means(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = Non_Local_Means(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Isodata(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_Isodata(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Li(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_Li(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Mean(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_Mean(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Minimum(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_Minimum(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Otsu(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_otsu(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Triangle(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_triangle(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_segBin_Yen(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_yen(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_line_detection(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        tam = array_imagem.shape
        maxt = int(max(tam[0], tam[1]))

        t = tk.simpledialog.askinteger("Input", f"Insira o threshold das linhas [0 - {maxt}]", parent=self.master, minvalue=0, maxvalue=maxt)
        if not t:
            tk.messagebox.showinfo("Erro", "Insira um valor de threshold!")
            return

        self.desenhar_imagemRGB(line_detection(array_imagem, t))

    def call_probabilistic_line_detection(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        tam = array_imagem.shape
        maxt = int(max(tam[0], tam[1]))

        t = tk.simpledialog.askinteger("Input", "Insira o threshold de votos", parent=self.master, minvalue=0)
        if t is None:
            tk.messagebox.showinfo("Erro", "Valor de threshold inválido!")
            return

        min_line_size = tk.simpledialog.askinteger("Input", f"Insira o tamanho mínimo das linhas [0 - {maxt}]", parent=self.master, minvalue=0, maxvalue=maxt)
        max_line_gap = tk.simpledialog.askinteger("Input", f"Insira o vão máximo de uma linha [0 - {maxt}]", parent=self.master, minvalue=0, maxvalue=maxt)

        nova_imagem = probabilistic_line_detection(array_imagem, t, min_line_size, max_line_gap)
        self.desenhar_imagemRGB(nova_imagem)

    def call_circle_detection(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        tam = array_imagem.shape
        minT = int(max(tam[0], tam[1])) // 2
        minR = tk.simpledialog.askinteger("Input", f"Insira o raio mínimo [0 - {minT}]", parent=self.master, minvalue=0, maxvalue=minT)
        if minR is None:
            tk.messagebox.showinfo("Erro", "Valor de raio inválido!")
            return
        maxR = tk.simpledialog.askinteger("Input", f"Insira o raio máximo [0 - {minT}]", parent=self.master, minvalue=0, maxvalue=minT)
        if maxR is None:
            tk.messagebox.showinfo("Erro", "Valor de raio inválido!")
            return
        nova_imagem = circle_detection(array_imagem, minR, maxR)
        self.desenhar_imagemRGB(nova_imagem)

    def call_component_detection(self, extraction):
        array_imagem = np.array(ImageTk.getimage(self.img))

        fig, ax = plt.subplots()

        if extraction:
            fig, ax = prepare_component(array_imagem, extraction)
        else:
            ax.imshow(prepare_component(array_imagem, extraction), cmap="nipy_spectral")
        plt.show()

    def call_segBin_Custon(self):
        limiar = tk.simpledialog.askinteger("Input", "Insira o limiar", parent=self.master, minvalue=0, maxvalue=255)
        if limiar == None:
            return
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = segBin_custom(array_imagem, limiar)
        self.desenhar_imagemRGB(nova_imagem)

    def call_detecBordas(self, func):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = func(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

    def call_detecBordas_Kernel(self):
        array_imagem = np.array(ImageTk.getimage(self.img))
        nova_imagem = detcBordas_kernel(array_imagem)
        self.desenhar_imagemRGB(nova_imagem)

root = tk.Tk()
app = Application(master=root)
app.master.title("Tophoshop")
app.master.minsize(1200, 900)
app.master.maxsize(1200, 900)
app.mainloop()