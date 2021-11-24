from typing import Final
import matplotlib
from skimage import data, restoration, filters, util, exposure, morphology
from scipy.ndimage import rotate
import numpy as np
from skimage.util.dtype import img_as_uint
import cv2.cv2 as cv
from random import choice, choices

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


def line_detection(img, ct1, ct2, t):
    border = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    if not ct1: ct1 = 50
    if not ct2: ct2 = 200
    border = cv.Canny(border, ct1, ct2, None, 3)

    linhas = cv.HoughLines(border, 1, np.pi / 180, t)

    result = np.copy(img)

    for linha in linhas:
        rho, theta = linha[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

        cv.line(result, pt1, pt2, (255, 0, 0), 3, cv.LINE_AA)
    return result


def probabilistic_line_detection(img, ct1, ct2, t, min_line_size, max_line_gap):
    border = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    if not ct1: ct1 = 50
    if not ct2: ct2 = 200
    border = cv.Canny(border, ct1, ct2, None, 3)

    linhas = cv.HoughLinesP(border, 1, np.pi / 180, t, None, min_line_size, max_line_gap)

    result = np.copy(img)

    for linha in linhas:
        x1, y1, x2, y2 = linha[0]

        cv.line(result, (x1, y2), (x2, y2), (255, 0, 0), 3, cv.LINE_AA)

    return result


def circle_detection(img, min_radius, max_radius):
    blur = cv.medianBlur(img, 5)
    blur = cv.cvtColor(blur, cv.COLOR_RGBA2GRAY)

    result = np.copy(img)

    circulos = cv.HoughCircles(blur,
                               cv.HOUGH_GRADIENT,
                               2,
                               min(result.shape[0], result.shape[1]) / 64,
                               param1=200,
                               param2=100,
                               minRadius=min_radius,
                               maxRadius=max_radius)
    if circulos is None: return img
    circulos = circulos.astype(int)

    for circulo in circulos[0]:
        centro = (circulo[0], circulo[1])

        cv.circle(result, centro, circulo[2], (255, 0, 0), 2)
        cv.circle(result, centro, 2, (0, 0, 255), 3)

    return result


def get_new_color(colors=[]):
    while True:
        color = choices(range(256), k=3)
        if color not in colors: break
    return color


def component_detection(img):
    current_label = 1
    colors = [[255, 255, 255]]
    colors.append(get_new_color([255, 255, 255]))
    queue = []

    result = np.zeros(img.shape)
    img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    img = level_reduction(img, 7)
    h, w = img.shape
    components = np.zeros(img.shape)

    # Passo 2
    for i in range(h):
        for j in range(w):
            if img[i][j] and not components[i][j]:
                components[i][j] = current_label
                result[i][j] = colors[current_label] + [255]
                queue.append((i, j))
            else:
                continue

            # Passo 3
            while len(queue):
                y, x = queue.pop(0)
                for ni in range(max(0, y - 1), min(h - 1, y + 1) + 1):
                    for nj in range(max(0, x - 1), min(w - 1, x + 1) + 1):
                        if img[ni][nj] and not components[ni][nj]:
                            components[ni][nj] = current_label
                            result[ni][nj] = colors[current_label] + [255]
                            queue.append((ni, nj))
            # Passo 4
            current_label += 1
            colors.append(get_new_color(colors))
    return result


def level_reduction(img, lvl):
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Lembrar de castar para inteiro para evitar um erro no qual a imagem simplesmente ignora as contas
            new_img[i][j] = int(img[i][j] / (2 ** lvl))
    return new_img
