import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from viktor import ViktorController
from viktor.parametrization import ViktorParametrization, GeoPointField, Text, MapSelectInteraction, SetParamsButton
from viktor.parametrization import NumberField, Tab
from viktor.result import SetParamsResult
from viktor.views import MapPolygon, MapResult, MapPoint, MapView, MapLine, MapLabel, Color

# Obtener la ruta absoluta del archivo
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script
db_file_path = os.path.join(base_dir, 'data', 'datos.h5')
regressor_file_path = os.path.join(base_dir, 'data', 'modelo_gradient_boosting.pkl')

class Predictor():
    def __init__(self, db_file_path="data/datos.h5", regressor_file_path="data/modelo_gradient_boosting.pkl") -> None:
        # Ruta del archivo donde se guarda el modelo
        regressor_file_pathh = 'data/modelo_gradient_boosting.pkl'

        # Verificar si el archivo existe
        if os.path.exists(regressor_file_path):
            print("Cargando el modelo existente desde archivo.")
            self.regressor = joblib.load(regressor_file_path)
        else:
        # Cargar datos
            self.data = pd.read_hdf(db_file_path, key='df')
            
            # Definimos las etiquetas
            self.labels = self.data['PreVivienda']

            # Creamos un DataFrame de entrenamiento removiendo
            # las columnas id y price
            self.train = self.data.drop(["PreVivienda"], axis=1)

            # Generamos conjuntos de entrenamiento y de prueba
            x_train , x_test , y_train , y_test = train_test_split(self.train,
                                                                self.labels,
                                                                test_size = 0.10,
                                                                random_state =2)

            # Generamos un regresor utilizando ensamblajes y Gradient Bootsting
            self.regressor = ensemble.GradientBoostingRegressor(n_estimators = 400,
                                                                max_depth = 5,
                                                                min_samples_split = 2,
                                                                learning_rate = 0.1,
                                                                loss = 'squared_error')
            
            # Entrenamos el regresor
            self.regressor.fit(x_train, y_train)

    def predict(self, input):
        self.prediction = self.regressor.predict(input)
        return self.prediction

# Crear predictor
predictor = Predictor(db_file_path, regressor_file_path)

# Carga de datos de viviendas
houses = pd.read_hdf(db_file_path, key='df')

class Parametrization(ViktorParametrization):
    intro = Text("""
# 🏠 App para estimación de costo de vivienda

En esta app puedes realizar la estimación del costo de una vivienda utilizando inteligencia artificial.
                 
De esta manera de puede hacer un análisis financiero de propiedades inmoviliarias.

**Selecciona el punto** correpondiente a la ubicación de la vivienda
    """)

    point = GeoPointField("Ubicación de nueva vivienda:")
    dormitorios = NumberField("Número de dormitorios:", default=2)
    sup_util = NumberField("Área útil [m2]: ", default=99.5)
    sup_ext = NumberField("Área exterior [m2]: ", default=10.2)
    sup_cons = NumberField("Área de construcción [m2]: ", default=118.5)
    total_viviendas= NumberField("Total de viviendas en promoción: ", default=40)
    piscina = NumberField("Tiene piscina (1: verdadero, 0: falso): ", default=1)
    jardin = NumberField("Tiene jardin comunitario (1: verdadero, 0: falso): ", default=0)
    parking = NumberField("Tiene parking (1: verdadero, 0: falso): ", default=0)
    trastero = NumberField("Tiene trastero (1: verdadero, 0: falso): ", default=0)
    parque_infantil = NumberField("Tiene parque infantil (1: verdadero, 0: falso): ", default=0)
    year = NumberField("Año de construcción:", default=2018)

class Controller(ViktorController):
    label = "Estimar costo de venta"
    parametrization = Parametrization

    @MapView("Análisis de ubicación", duration_guess=1)
    def generate_map(self, params, **kwargs):
        house = houses.iloc[0]
        lat = house["Latitud"]
        long = house["Longitud"]

        # Crear punto en mapa utilizando las coordenadas
        some_point = MapPoint(lat, long, description='01', identifier='01')

        features = []
        # Crear puntos desde datos
        for i in range(30):
            house = houses.iloc[i]
            lat = house["Latitud"]
            long = house["Longitud"]
            description = f"Precio: {house['PreVivienda']} €\n"
            description += f"Latitud: {house["Latitud"]}\n"
            description += f"Longitud: {house["Longitud"]}\n"
            description += f"Núm. Dormitorios: {house['Dormitorios']}\n"
            description += f"Superficie útil: {house['SupUtil']}\n"
            description += f"Superficie exterior: {house["SupExt"]}\n"
            description += f"Superficie construida: {house["SupCons"]}\n"
            description += f"Viviendas totales en promoción: {house["TotalViviendas"]}\n"
            description += f"Tiene piscina: {house["piscina"]}\n"
            description += f"Tiene jardin: {house["jardin"]}\n"
            description += f"Tiene parking {house["parking"]}\n"
            description += f"Tiene trastero: {house["trastero"]}\n"
            description += f"Tiene parque infantil: {house["parque_infantil"]}\n"
            description += f"Año de construcción: {house["Año"]}"

            point_i = MapPoint(lat, long, description=description, identifier=str(i))
            features.append(point_i)
        
        # Obtener punto desde parámetros de entrada y agregarlo 
        # a las características si existe
        if params.point:
            input_point = MapPoint.from_geo_point(params.point)
            input_lat = input_point.lat
            input_long = input_point.lon
            input_data = [params.dormitorios,
                          params.sup_util,
                          params.sup_ext,
                          params.sup_cons,
                          params.year,
                          input_lat,
                          input_long,
                          params.total_viviendas,
                          params.piscina,
                          params.jardin,
                          params.parking,
                          params.trastero,
                          params.parque_infantil]

            price = predictor.predict([input_data])[0]
            print(price)
            prediction_point = MapPoint(input_lat,
                                        input_long,
                                        icon="cross",
                                        description=f"Precio estimado: {price:.2f} €")
            features.append(prediction_point)

        return MapResult(features)

