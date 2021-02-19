import sys
from reconocimiento import reconocimiento_facial,registrar_coordenadas_usuario

print("Inicio del Programa")
# print(reconocimiento_facial("ImagesLogin/demo.jpg"))
# print(reconocimiento_facial("ImagesLogin/demo2.jpg"))
# print(reconocimiento_facial("ImagesLogin/demo3.jpg"))
# print(reconocimiento_facial("ImagesLogin/demo4.jpg"))
# print(reconocimiento_facial("ImagesLogin/demo5.jpg"))
#entrenamiento("ImagesLogin/james.jpg",3)
# file = open("ImagesLogin/demo.jpg", "r")

#print(registrar_coordenadas_usuario("ImagesLogin/yankee.jpg",17))

print(reconocimiento_facial("ImagesLogin/photo.png"))
print("Fin del Programa")

sys.exit(0)