# Domina Python: Scikit-learn

Este es el repositorio del curso de LinkedIn Learning `Domina Python: Scikit-learn`. El curso completo está disponible en [LinkedIn Learning][lil-course-url].

![Nombre completo del curso][lil-thumbnail-url] 

Consulta el archivo Readme en la rama main para obtener instrucciones e información actualizadas.

Ingresa al mundo del aprendizaje automático y la ciencia de datos con este curso especializado. Aprende a usar la potente biblioteca Scikit-learn en Python para desarrollar, evaluar y seleccionar modelos de aprendizaje automático. Descubre cómo abordar desafíos de predicción de datos al dominar herramientas clave y conceptos como selección de características, aprendizaje supervisado y no supervisado, y evaluación de modelos. Este curso te prepara para aplicar el aprendizaje automático en situaciones reales y tomar decisiones informadas.

## Instrucciones

Este repositorio tiene directorios para cada uno de los capítulos del curso.

## Directorios

Las directorios están estructuradas para corresponder a los vídeos del curso. La convención de nomenclatura del directorio es c# donde c corresponde a capítulo y # al número del capítulo, para los archivos la convención es c#v#, donde c corresponde a capítulo, v corresponde a video y # corresponde al número del capítulo y del video respectivamente. Por ejemplo, el directorio denominada c2_ corresponde al segundo capítulo y el archivo que se encuentra en este directorio iniciando con el nombre c2v03_ corresponde al tercer vídeo del segundo capítulo.

## Instalación

1. Para utilizar estos archivos de ejercicios, debes tener descargado lo siguiente:
   - Python
   - Editor de código como VS Code o PyCharm
2. Clona este repositorio en tu máquina local usando la Terminal (macOS) o CMD (Windows), o una herramienta GUI como SourceTree.
3. Crea un ambiente virtual de Python, puedes hacerlo con virtualenv usando los comandos

		pip install virtualenv
		virtualenv <reemplazar por nombre del ambiente>

4. Instala las librerías con el comando

		pip install -r requirements.txt

5.  Corre cada archivo con el comando

		python <nombre_archivo>

### Docente

**Ana María Pinto**

Echa un vistazo a mis otros cursos en [LinkedIn Learning](https://www.linkedin.com/learning/instructors/ana-maria-pinto).

[0]: # (Replace these placeholder URLs with actual course URLs)
[lil-course-url]: https://www.linkedin.com/learning/domina-python-scikit-learn
[lil-thumbnail-url]: https://media.licdn.com/dms/image/D4E0DAQHg21EVGXtkCQ/learning-public-crop_675_1200/0/1708518857043?e=2147483647&v=beta&t=LpRG2lQS-nrOl_mN3RndZTTo26s-HthTy5DSi_e0lxs

[1]: # (End of ES-Instruction ###############################################################################################)


### Informacion básica Libreria 

***  Informacion básica Libreria scikit-learn *** 
Características principales:

	Modelos de aprendizaje supervisado: regresión (lineal, logística), clasificación (SVM, KNN, árboles de decisión, Random Forest, etc.).

	Modelos no supervisados: clustering (K-means, DBSCAN), reducción de dimensionalidad (PCA, t-SNE).

	Preprocesamiento de datos: escalado, codificación de variables categóricas, imputación de valores faltantes, selección de características.

	Evaluación de modelos: validación cruzada, métricas de rendimiento (precisión, recall, F1-score, ROC).

	Pipelines: para encadenar transformaciones y modelos en flujos de trabajo reproducibles.

	Carga de datasets de prueba: incluye datasets integrados como Iris, dígitos, Boston (obsoleto), etc.

Ventajas:

	API coherente y bien documentada.
	Muy buena integración con el ecosistema científico de Python.

	Ideal para prototipado rápido y aplicaciones de producción ligeras.

Limitaciones:

	No está diseñada para deep learning (para eso se usan TensorFlow o PyTorch).
	No está optimizada para datasets extremadamente grandes (usa procesamiento en memoria).