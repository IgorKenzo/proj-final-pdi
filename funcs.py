from skimage import data, feature, filters, transform
import numpy as np
import skimage
from skimage.util.dtype import img_as_uint

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

def rotate_img(imagem, angulo):    
    a = transform.rotate(imagem, angle=angulo)
    return img_as_uint(a)

