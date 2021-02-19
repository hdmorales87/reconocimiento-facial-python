import face_recognition
from imutils import paths
import imutils
import pickle
import time
import cv2
import os
import json
import numpy
import pymysql

# libreria que contiene las funciones para reconocimiento de voz

'''
    registrar_coordenadas_usuario : Registra la coordenadas del usuario para su posterior reconocimiento

    Parameters
    ----------

    Returns
    -------
    void
        Mensaje de exito o de error

''' 

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='reconocimiento',
                             cursorclass=pymysql.cursors.DictCursor)  

def registrar_coordenadas_usuario(file,id_user):
    try: 
        # Cargar la imagen y convertirla de BGR(escala de grises)
        # a RGB
        image = cv2.imread(file)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Usar Face_recognition para detectar rostros
        boxes = face_recognition.face_locations(rgb)

        # Codificar el encrustamiento facial para el rostro
        encodings = face_recognition.face_encodings(rgb, boxes) 
        
        if len(encodings) > 0: 
            # Guardar en JSON las coordenadas del usuario       
            json_coords = numpy_to_json(encodings[0]) 

            # Registrar en la base de datos las coordenadas del usuario
            with connection:

                with connection.cursor() as cursor:
                    # Actualizar la información del usuario
                    sql = "UPDATE `users` SET coordenadas_rostro=%s WHERE id=%s"
                    cursor.execute(sql, (json_coords,id_user))

                # connection is not autocommit by default. So you must commit to save            
                connection.commit()        

            #Retornar mensaje de exito
            return "success"
        else:
            #Retornar mensaje de error
            return "foto_sin_rostro"
    except Exception as e:
        return "Ha ocurrido un error"+ str(e)
 
'''
    reconocimiento_facial : Realiza el reconocimiento facial con base en el entrenamiento almacenado

    Parameters
    ----------
    String
        Ruta del archivo a examinar

    Returns
    -------    
        id de persona identificada, de lo contrario enviará no_registrado

'''  
def reconocimiento_facial(file):
    try:
        # Obtener archivo xml haarcascade  
        cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

        # Cargar el archivo harcaascade en el clasificador en cascada
        faceCascade = cv2.CascadeClassifier(cascPathface)        

        # Definir arreglos
        knownCoords = []
        knownUsers = [] 

        with connection.cursor() as cursor:
            # Traer todas las coordenadas registradas
            sql = "SELECT `coordenadas_rostro`, `id` FROM `users`"
            cursor.execute(sql)
            rows = cursor.fetchall()            

            for row in rows:
                # Valida que el objeto no esté vacio
                if row['coordenadas_rostro'] is not None:  
                    face_coords = json_to_numpy(row['coordenadas_rostro'])
                    id_user = row['id']
                    knownCoords.append(face_coords)
                    knownUsers.append(id_user)

            data = {"encodings": knownCoords, "users": knownUsers}   

            # Obtener la imagen a verificar a partir de un archivo
            image = face_recognition.load_image_file(file)

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Encontrar rostros o ojos en la imagen
            faces = faceCascade.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(60, 60),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

            # Convertir la imagen entrante de BGR(Escala de Grises) to RGB(Color) 
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Los rostros incrustados en la imagen
            encodings = face_recognition.face_encodings(rgb)  

            id_user = "foto_sin_rostro"          
            
            if len(encodings) > 0:
                # Comparar coordenadas del rostro de la imagen con las codificaciones del array data["encodings"]
                # Coincidencias contienen array con valores booleandos (True o False) y es True si es una estrecha coincidencia 
                matches = face_recognition.compare_faces(data["encodings"], encodings[0])
                
                # Poner nombre desconocido si no hay coincidencia
                id_user = "no_registrado"                                 

                # Verificar si hay coincidencias
                if True in matches:
                    # Encontrar los indices donde hay True y almacenarlos
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {} 

                    # Recorrer los indices y contar cada rostro reconocido
                    for i in matchedIdxs:
                        # Chequear las personas con base en el indice asociado en el archivo de rostros
                        id_user = data["users"][i]
                        
                        # Incrementar cuenta para la persona que ya tenemos
                        counts[id_user] = counts.get(id_user, 0) + 1

                    # Poner la persona que tenga la mayor cuenta
                    id_user = max(counts, key=counts.get)
            
            return id_user  
    except Exception as e:
        return("Ha ocurrido un error"+ str(e))  

def numpy_to_json(n_array):
    n_list = n_array.tolist()
    j_array = json.dumps(n_list) 
    return j_array

def json_to_numpy(j_data):
    j_array = json.loads(j_data)
    n_array = numpy.array(j_array)
    return n_array  