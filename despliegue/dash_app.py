import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import base64
import io
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go
import dill

#px.plot

# Leer el archivo CSV
data_df = pd.read_csv("../docs/descriptive_data.csv")

#Cargar modelo
model = dill.load(open("../despliegue/model.pkl", "rb"))
#Coeficientes
coeficientes = model.named_steps["model"].coef_
# Intercepto del modelo para el Saber 11
intercepto_saber11 = model.named_steps["model"].intercept_

# Mapear los nombres de los coeficientes a nombres más amigables para el usuario
nombres_coeficientes = {
    "FAMI_TIENEAUTOMOVIL_Si": "Tiene Automóvil",
    "FAMI_TIENECOMPUTADOR_Si": "Tiene Computadora",
    "FAMI_TIENEINTERNET_Si": "Tiene Internet",
    "FAMI_TIENELAVADORA_Si": "Tiene Lavadora",
    "COLE_AREA_UBICACION_URBANO": "Área Urbana",
    "FAMI_EDUCACIONMADRE_no_presenta": "Educación Madre: No presenta",
    "FAMI_EDUCACIONMADRE_basica": "Educación Madre: Básica",
    "FAMI_EDUCACIONMADRE_media": "Educación Madre: Media",
    "FAMI_EDUCACIONPADRE_no_presenta": "Educación Padre: No presenta",
    "FAMI_EDUCACIONPADRE_basica": "Educación Padre: Básica",
    "FAMI_EDUCACIONPADRE_media": "Educación Padre: Media",
    "FAMI_ESTRATOVIVIENDA_Estrato_2": "Estrato Vivienda: 2",
    "FAMI_ESTRATOVIVIENDA_Estrato_3": "Estrato Vivienda: 3",
    "FAMI_ESTRATOVIVIENDA_Estrato_4": "Estrato Vivienda: 4",
    "FAMI_ESTRATOVIVIENDA_Estrato_5": "Estrato Vivienda: 5",
    "FAMI_ESTRATOVIVIENDA_Estrato_6": "Estrato Vivienda: 6"
}

#Lista nombre coeficientes
names_coef = list(nombres_coeficientes.keys())
#Construccion tupla coeficientes
coeficientes_saber11 = {}
for i in range(len(names_coef)):
    coeficientes_saber11[names_coef[i]] = coeficientes[0][i]

# Ordenar los coeficientes de menor a mayor
coeficientes_ordenados = sorted(coeficientes_saber11.items(), key=lambda x: x[1])
# Crear una lista para almacenar los divs generados dinámicamente
divs = []
# Iterar sobre los valores de la serie datos_description
for nombre_coeficiente, valor_coeficiente in coeficientes_ordenados:
    # Calcular la intensidad del color azul para el coeficiente
    color_azul = 400 - int((valor_coeficiente - min(coeficientes_saber11.values())) / (max(coeficientes_saber11.values()) - min(coeficientes_saber11.values())) * 350)

    # Definir el estilo del div para el coeficiente
    estilo = {
        'font-family': 'Calibri',
        'font-size': '16px',
        'text-align': 'center',
        'margin': '5px auto',  # Ajustar margen para centrar en una columna
        'box-shadow': '2px 2px',
        'border-radius': '5px',
        'padding': '10px',  # Aumentar el padding para mejor visualización
        'width': '60%',  # Aumentar el ancho para mejor visualización
        'color': 'white',  # Texto en blanco,
        'background': f'rgba(0, 0, {color_azul}, 1)'  # Color de fondo azul con intensidad variable
    }

    # Crear el contenido del div
    contenido = f"{nombres_coeficientes[nombre_coeficiente]}: {valor_coeficiente:.2f}"

    # Agregar el div a la lista divs
    div = html.Div(contenido, style=estilo)
    divs.append(div)

# Dropdown
dropdown_options = [
    {"label": "Tiene Automóvil", "value": "FAMI_TIENEAUTOMOVIL"},
    {"label": "Tiene Computadora", "value": "FAMI_TIENECOMPUTADOR"},
    {"label": "Tiene Internet", "value": "FAMI_TIENEINTERNET"},
    {"label": "Tiene Lavadora", "value": "FAMI_TIENELAVADORA"},
    {"label": "Área Urbana", "value": "COLE_AREA_UBICACION"},
    {"label": "Estrato Vivienda", "value": "FAMI_ESTRATOVIVIENDA"},
    {"label": "Cuartos en el Hogar", "value": "FAMI_CUARTOSHOGAR"},
    {"label": "Educación Madre", "value": "FAMI_EDUCACIONMADRE"},
    {"label": "Educación Padre", "value": "FAMI_EDUCACIONPADRE"}
]
# Definir los nombres correspondientes a las opciones del dropdown
nombres_coeficientes = {
    "FAMI_TIENEAUTOMOVIL": "Tiene Automóvil",
    "FAMI_TIENECOMPUTADOR": "Tiene Computadora",
    "FAMI_TIENEINTERNET": "Tiene Internet",
    "FAMI_TIENELAVADORA": "Tiene Lavadora",
    "COLE_AREA_UBICACION": "Área Urbana",
    "FAMI_ESTRATOVIVIENDA": "Estrato Vivienda",
    "FAMI_CUARTOSHOGAR": "Cuartos en el Hogar",
    "FAMI_EDUCACIONMADRE": "Educación Madre",
    "FAMI_EDUCACIONPADRE": "Educación Padre"}

## Función para calcular la predicción del Saber 11 y el intervalo de confianza
def predecir_saber11(df, coeficientes, intercepto):
    # Filtrar las columnas del DataFrame para incluir solo aquellas que tienen un coeficiente asociado
    columnas_filtradas = [col for col in df.columns if col in coeficientes]
    # Filtrar el DataFrame para que solo incluya las columnas relevantes
    df_filtrado = df[columnas_filtradas]
    # Convertir el DataFrame filtrado a un array de numpy
    valores_array = df_filtrado.values
    # Convertir los coeficientes a un array de numpy y asegurarnos de que estén en el mismo orden que las columnas del DataFrame
    coeficientes_array = np.array([coeficientes[col] for col in df_filtrado.columns])
    # Realizar el producto punto entre los valores del DataFrame y los coeficientes
    predicciones = np.dot(valores_array, coeficientes_array) + intercepto
    return predicciones

# Función para cargar el archivo CSV y mostrar la predicción del Saber 11 y el intervalo de confianza
# Luego, modifica la función parse_contents para corregir la creación de la gráfica de campana
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),index_col=0)
    # Realizar la predicción utilizando los datos del archivo
    prediccion_saber11 = predecir_saber11(df, coeficientes_saber11, intercepto_saber11)
    # Crear la figura de la gráfica de campana
    bell_curve = px.histogram(x=prediccion_saber11, nbins=25)
    bell_curve.update_layout(title='Predicción del Saber 11 y Intervalo de Confianza',
                             xaxis_title='Puntaje predicho del Saber 11',
                             yaxis_title='Densidad de probabilidad',
                             plot_bgcolor='rgba(0,0,0,0)', # Establece el color de fondo del gráfico como transparente
                             paper_bgcolor='white' # Establece el color de fondo del área de la gráfica como blanco
                             )
    #Calcular metricas importantes
    #Media
    media = np.mean(prediccion_saber11)
    #Intervalo de prediccion al 95%
    n = len(prediccion_saber11)
    std_err = np.std(prediccion_saber11)
    h = std_err * t.ppf((0.975) / 2, n - 1)
    #Intervalo de confianza
    lower_bound = media - h
    upper_bound = media + h
    #Quartiles
    q1 = np.percentile(prediccion_saber11, 25)
    q2 = np.percentile(prediccion_saber11, 50)
    q3 = np.percentile(prediccion_saber11, 75)
    #Limites de aotliers y cuantos datos hay en outliers
    iqr = q3 - q1
    lower_bound_outliers = q1 - 1.5 * iqr
    upper_bound_outliers = q3 + 1.5 * iqr
    outliers_lower = round(np.count_nonzero(prediccion_saber11 < lower_bound_outliers)/n, 2)*100
    outliers_upper = round(np.count_nonzero(prediccion_saber11 > upper_bound_outliers)/n, 2)*100

    #Contrucción de tablas
    stylr_td = {'text-align': 'center'}
    style_tr = {'padding': '10px'}
    tabla_media =  [
        html.Tr([
            html.Th('Media', style={'width': '40%'}), html.Th('Intervalo de Confianza (95%)', style={'width': '60%'})
        ]),
        html.Tr([
            html.Td(f'{media:.2f}', style = stylr_td), html.Td(f'[{lower_bound:.2f}, {upper_bound:.2f}]', style = stylr_td)
        ], style = style_tr)
    ]

    tabla_quartiles =  [
        html.Tr([
            html.Th('Quartile 1', style={'width': '25%'}), html.Th('Quartile 2', style={'width': '25%'}), html.Th('Quartile 3', style={'width': '25%'})
        ]),
        html.Tr([
            html.Td(f'{q1:.2f}', style = stylr_td), html.Td(f'{q2:.2f}', style = stylr_td), html.Td(f'{q3:.2f}', style = stylr_td)
        ], style = style_tr)
    ]

    tabla_outliers =  [
        html.Tr([
            html.Th('Limite Inferior', style={'width': '25%'}), html.Th('Limite Superior', style={'width': '25%'}), 
            html.Th('% Outliers Inferiores', style={'width': '25%'}), html.Th('% Outliers Superiores', style={'width': '25%'})
        ]),
        html.Tr([
            html.Td(f'{lower_bound_outliers:.2f}', style = stylr_td), html.Td(f'{upper_bound_outliers:.2f}', style = stylr_td),
            html.Td(f'{outliers_lower}%', style = stylr_td), html.Td(f'{outliers_upper}%', style = stylr_td)
        ], style = style_tr)
    ]
    #Estilo gráfica
    graph_style = {'margin': '20px','box-shadow': '2px 2px 5px grey', 'border-radius': '5px', 'padding': '5px','width': '90%'}
    return html.Div([
                html.Div([
                    dcc.Graph(figure=bell_curve)
                ],style=graph_style),
                html.Div([
                    html.Table(tabla_media,style=graph_style),
                    html.Table(tabla_quartiles,style=graph_style),
                    html.Table(tabla_outliers,style=graph_style)
                ]),
            ],style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'} )


# Definir el color azul personalizado para el fondo y las gráficas
color_azul_personalizado = 'white'

# Inicializar la aplicación de Dash
app = dash.Dash(__name__)
# LAYOUT
app.layout = html.Div([
    html.Div(
        style={'backgroundColor': color_azul_personalizado},
        children=[
            html.H1("Análisis de Datos - Saber 11- Dep_Meta", style={'textAlign': 'center', 'color': 'black'}),
            html.Label("Selecciona una variable:", style={'color': 'black'}),
            dcc.Dropdown(
                id='variable-dropdown',
                options=dropdown_options,
                value=None
            ),

        ]
    ),
    html.Div([
        html.Div(divs, id='coeficientes-container', style={'width': '30%', 'flex-direction': 'column', 'justify-content': 'center'}),
        html.Div([
            dcc.Graph(id='box-plot', figure={}, style={'width': '50%'}),
            dcc.Graph(id='bar-chart', figure={}, style={'width': '50%'})
        ], id='graphs-container', style={'width': '90%', 'display': 'flex', 'flex-direction': 'row'})
    ], style={'display': 'flex', 'width': '100%', 'justify-content': 'center'}),

    html.Div(
        style={'backgroundColor': 'white'},
        children=[  # Mover 'children' dentro del mismo html.Div
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Arrastra y suelta o ',
                    html.A('Selecciona el archivo CSV')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ]
    )
])
# Callback para actualizar los gráficos
@app.callback(
    [Output('box-plot', 'figure'),
     Output('bar-chart', 'figure')],
    [Input('variable-dropdown', 'value')]
)
def update_graphs(selected_variable):
    if not selected_variable:
        return {}, {}

    # Box plot
    box_fig = px.box(data_df, x=selected_variable, y='PUNT_GLOBAL', title=f"Relación entre Puntuación Global y {nombres_coeficientes[selected_variable]}")

    # Bar chart
    bar_fig = px.bar(data_df, x=selected_variable, title=f"Conteo de {nombres_coeficientes[selected_variable]}")

    # Establecer el color azul personalizado en las gráficas
    box_fig.update_layout(plot_bgcolor=color_azul_personalizado, paper_bgcolor=color_azul_personalizado)
    bar_fig.update_layout(plot_bgcolor="White", paper_bgcolor="White")
    return box_fig, bar_fig


# Callback para cargar el archivo y mostrar la predicción del Saber 11 y el intervalo de confianza
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        children = [
            parse_contents(contents, filename)
        ]
        return children
# Ejecutar la aplicación de Dash

if __name__ == '__main__':
  app.run_server(host = "0.0.0.0", debug=True, port=8050)