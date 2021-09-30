from skimage import data, feature, filters
import numpy as np

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
