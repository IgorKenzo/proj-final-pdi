from typing import Final
import matplotlib.pyplot as plt
from skimage import data, restoration, filters, util, exposure, morphology
from scipy.ndimage import rotate
import numpy as np
from skimage.util.dtype import img_as_uint
import cv2

def simple_blur(imagem, tamanho_kernel):
    imagemBlur = np.zeros(imagem.shape, dtype=np.uint8)
    imagemBlur[:,:, 3] = 255

    if tamanho_kernel % 2 == 0:
        deslocamentoEsquerda = tamanho_kernel//2
        deslocamentoDireita = tamanho_kernel//2
    else:
        deslocamentoEsquerda = tamanho_kernel//2
        deslocamentoDireita = (tamanho_kernel//2)+1

    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            for camada in range(3):
                vizinhos = 0
                count = 0
                for i in range(x-deslocamentoEsquerda,x+deslocamentoDireita):
                    for j in range(y-deslocamentoEsquerda,y+deslocamentoDireita):
                        if i >= 0 and i < imagem.shape[0] and j >= 0 and j < imagem.shape[1]:
                            vizinhos += imagem[i][j][camada]
                            count += 1
                media = int(vizinhos / count)
                imagemBlur[x][y][camada] = media
    
    # print(imagemBlur[0][0])
    return imagemBlur

def mosaic(imagem, tamanho_kernel):
    imagemBlur = np.zeros(imagem.shape, dtype=np.uint8)
    imagemBlur[:,:, 3] = 255

    for x in range(0, imagem.shape[0], tamanho_kernel):
        for y in range(0, imagem.shape[1], tamanho_kernel):
            for camada in range(3):
                if x < imagem.shape[0] and y < imagem.shape[1]:
                    vizinhos = 0
                    count = 0
                    for i in range(x,x+tamanho_kernel):
                        for j in range(y,y+tamanho_kernel):    
                            if i >= 0 and i < imagem.shape[0] and j >= 0 and j < imagem.shape[1]:       
                                vizinhos += imagem[i][j][camada]
                                count += 1    
                        
                    media = int(vizinhos / count)

                    for i in range(x,x+tamanho_kernel):
                        for j in range(y,y+tamanho_kernel):
                            if i >= 0 and i < imagem.shape[0] and j >= 0 and j < imagem.shape[1]:
                                imagemBlur[i][j][camada] = media
    
    # print(imagemBlur[0][0])
    return imagemBlur

def match_operation(imagem1, imagem2):
    match = exposure.match_histograms(imagem1, imagem2, multichannel=True)
    return img_as_uint(match)

def rotate_img(imagem, angulo):    
    a = rotate(imagem, angle=angulo, reshape=False)
    return img_as_uint(a)

def flip_hor(imagem):
    imgFlip = np.zeros(imagem.shape)
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            imgFlip[x, -y] = imagem[x, y]
    return imgFlip

def flip_ver(imagem):
    imgFlip = np.zeros(imagem.shape)
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            imgFlip[-x, y] = imagem[x, y]
    return imgFlip

def filtroGaussiano(imagem, sigma):
    tamanhoFiltro = 2 * int(4*sigma-0.5) + 1
    filtro = np.zeros((tamanhoFiltro, tamanhoFiltro), np.float32)

    h = tamanhoFiltro // 2
    w = tamanhoFiltro // 2

    for i in range(-h, h+1):
        for j in range(-w, w+1):
            x1 = 2 * np.pi * (sigma**2)
            x2 = np.exp(-(i**2 + j**2)/(2 * sigma**2))
            filtro[i+h][j+w] = (1/x1)*x2

    filtrada = np.zeros_like(imagem, np.float32)
    filtrada[:, :] = convolucao(imagem[:,:], filtro)

    return filtrada.astype(np.uint8)


def convolucao(imagem, kernel):
    himagem = imagem.shape[0]
    wimagem = imagem.shape[1]

    hkernel = kernel.shape[0]
    wkernel = kernel.shape[1]

    h = hkernel//2
    w = wkernel//2

    imagemconv = np.zeros(imagem.shape, np.float32)

    for linha in range(himagem):
        for coluna in range(wimagem):
            soma = 0
            for linhak in range(hkernel):
                for colunak in range(wkernel):
                    if linha+linhak-h >= 0 and linha+linhak-h < himagem and coluna+colunak-w >= 0 and coluna+colunak-w < wimagem:
                        soma += kernel[linhak][colunak] * imagem[linha+linhak-h][coluna+colunak-w]
            imagemconv[linha][coluna] = soma

    return imagemconv

def gaussian_filter_default(imagem, sigma):    
    return filters.gaussian(imagem, sigma, preserve_range=True, multichannel=True)

def salt_and_peper(imagem):
    a = util.random_noise(imagem, mode="s&p", salt_vs_pepper=0.5)
    return a*255

def unsharp_mask(imagem):
    nova_imagem = np.zeros(imagem.shape, np.uint8)
    nova_imagem[:, :, 3] = 255

    imagem = imagem[:, :, :3]
    imagem = imagem / 255
    borrada = filters.gaussian(imagem, sigma=2, mode="constant", cval=0, multichannel=True)

    subtraida = imagem - borrada

    final = imagem + (2 * subtraida)
    
    nova_imagem[:, :, :3] = final*255

    print(nova_imagem)
    return nova_imagem

def median_filter(imagem):
    disco = morphology.disk(1)
    limpa = filters.median(imagem[:, :, 0], disco, mode='constant', cval=0)
    
    nova_imagem = np.zeros(imagem.shape, np.uint8)
    nova_imagem[:, :, 0] = limpa
    nova_imagem[:, :, 1] = limpa
    nova_imagem[:, :, 2] = limpa
    nova_imagem[:, :, 3] = 255
    
    return nova_imagem

def noise(imagem, sigma):
    sigma = sigma/100
    img = util.random_noise(imagem, var=sigma**2)
    img *= 255
    img[:, :, 3] = 255
    return img

def Non_Local_Means(imagem):
    sigmaEstimado = np.mean(restoration.estimate_sigma(imagem, multichannel=True))

    limpa = restoration.denoise_nl_means(imagem, h=0.6 * sigmaEstimado, sigma=sigmaEstimado, patch_size=5, patch_distance=6, multichannel=True, preserve_range=True)
    return limpa

def segmentacao_Binaria(imagem, limiar):
    classificada = np.ones(imagem.shape)
    classificada *= 255

    for linha in range(imagem.shape[0]):
        for coluna in range(imagem.shape[0]):
            if imagem[linha][coluna][0] < limiar:
                classificada[linha][coluna][0] = 0

    return classificada

def segBin_Isodata(imagem):
    limiar = filters.threshold_isodata(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_Li(imagem):
    limiar = filters.threshold_li(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_Mean(imagem):
    limiar = filters.threshold_mean(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_Minimum(imagem):
    limiar = filters.threshold_minimum(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_otsu(imagem):
    limiar = filters.threshold_otsu(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_triangle(imagem):
    limiar = filters.threshold_triangle(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def segBin_yen(imagem):
    limiar = filters.threshold_yen(imagem)
    img = segmentacao_Binaria(imagem, limiar= limiar)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 3] = 255

    return img

def fourrierTransform(img, tipo):
    # dft = np.fft.fft2(img)
    # shift_dft = np.fft.fftshift(dft)
    # magShift = np.log(1+abs(shift_dft))
    dft2 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    shift = np.fft.fftshift(dft2)
    magnitude2 = 20 * np.log(cv2.magnitude(shift[:,:,0], shift[:,:,1]))
    lin, col = img.shape

    mascara = None

    if tipo == "alta":
        mascara = np.ones((lin, col, 2), dtype=np.uint8)
    else:
        mascara = np.zeros((lin, col, 2), dtype=np.uint8)

    r = 50

    centro = [lin//2, col//2]

    x, y = np.ogrid[:lin, :col]

    area_mascara = (x - centro[0]) ** 2 + (y - centro[1]) ** 2 <= r ** 2

    mascara[area_mascara] = 0 if tipo == "alta" else 1

    mascara_shift2 = shift * mascara

    mag_masc_shift = np.log(1+abs(mascara_shift2))
    ma = cv2.magnitude(mag_masc_shift[:,:,0], mag_masc_shift[:,:,1])
    print(mascara)

    inv_shift = np.fft.ifftshift(mascara_shift2)
    inv_dft = cv2.idft(inv_shift)
    mag2 = cv2.magnitude(inv_dft[:,:,0], inv_dft[:,:,1])

    return magnitude2, mag_masc_shift, mag2 #inv_dft, mag2