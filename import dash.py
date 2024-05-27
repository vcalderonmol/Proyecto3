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

# Leer el archivo CSV
data_df = pd.read_csv("Filtered_Resultados_Saber_11.csv")


# Definir los coeficientes y sus nombres
coeficientes = {
    "FAMI_TIENEAUTOMOVIL_Si": 2.65,
    "FAMI_TIENECOMPUTADOR_Si": 7.38,
    "FAMI_TIENEINTERNET_Si": 3.98,
    "FAMI_TIENELAVADORA_Si": 2.64,
    "COLE_AREA_UBICACION_URBANO": -3.06,
    "FAMI_EDUCACIONMADRE_no presenta": -22.39,
    "FAMI_EDUCACIONMADRE_básica": -20.38,
    "FAMI_EDUCACIONMADRE_media": -13.54,
    "FAMI_EDUCACIONPADRE_no presenta": -14.97,
    "FAMI_EDUCACIONPADRE_básica": -17.86,
    "FAMI_EDUCACIONPADRE_media": -13.01,
    "FAMI_ESTRATOVIVIENDA_Estrato 2": -0.70,
    "FAMI_ESTRATOVIVIENDA_Estrato 3": 2.98,
    "FAMI_ESTRATOVIVIENDA_Estrato 4": 5.83,
    "FAMI_ESTRATOVIVIENDA_Estrato 5": 12.28,
    "FAMI_ESTRATOVIVIENDA_Estrato 6": 14.40
}

# Mapear los nombres de los coeficientes a nombres más amigables para el usuario
nombres_coeficientes = {
    "FAMI_TIENEAUTOMOVIL_Si": "Tiene Automóvil",
    "FAMI_TIENECOMPUTADOR_Si": "Tiene Computadora",
    "FAMI_TIENEINTERNET_Si": "Tiene Internet",
    "FAMI_TIENELAVADORA_Si": "Tiene Lavadora",
    "COLE_AREA_UBICACION_URBANO": "Área Urbana",
    "FAMI_EDUCACIONMADRE_no presenta": "Educación Madre: No presenta",
    "FAMI_EDUCACIONMADRE_básica": "Educación Madre: Básica",
    "FAMI_EDUCACIONMADRE_media": "Educación Madre: Media",
    "FAMI_EDUCACIONPADRE_no presenta": "Educación Padre: No presenta",
    "FAMI_EDUCACIONPADRE_básica": "Educación Padre: Básica",
    "FAMI_EDUCACIONPADRE_media": "Educación Padre: Media",
    "FAMI_ESTRATOVIVIENDA_Estrato 2": "Estrato Vivienda: 2",
    "FAMI_ESTRATOVIVIENDA_Estrato 3": "Estrato Vivienda: 3",
    "FAMI_ESTRATOVIVIENDA_Estrato 4": "Estrato Vivienda: 4",
    "FAMI_ESTRATOVIVIENDA_Estrato 5": "Estrato Vivienda: 5",
    "FAMI_ESTRATOVIVIENDA_Estrato 6": "Estrato Vivienda: 6"
}

# Ordenar los coeficientes de menor a mayor
coeficientes_ordenados = sorted(coeficientes.items(), key=lambda x: x[1])

# Coeficientes del modelo para el Saber 11
coeficientes_saber11 = {
    "FAMI_TIENEAUTOMOVIL_Si": 2.6455104558916567,
    "FAMI_TIENECOMPUTADOR_Si": 7.376463853122136,
    "FAMI_TIENEINTERNET_Si": 3.9831811268082116,
    "FAMI_TIENELAVADORA_Si": 2.6413986271667462,
    "COLE_AREA_UBICACION_URBANO": -3.0611699748047303,
    "FAMI_EDUCACIONMADRE_no_presenta": -22.38813264553304,
    "FAMI_EDUCACIONMADRE_basica": -20.381362870257238,
    "FAMI_EDUCACIONMADRE_media": -13.54344769863283,
    "FAMI_EDUCACIONPADRE_no_presenta": -14.966785366297767,
    "FAMI_EDUCACIONPADRE_basica": -17.86398509154922,
    "FAMI_EDUCACIONPADRE_media": -13.008383875099819,
    "FAMI_ESTRATOVIVIENDA_Estrato_2": -0.70441715439153,
    "FAMI_ESTRATOVIVIENDA_Estrato_3": 2.9848453835640014,
    "FAMI_ESTRATOVIVIENDA_Estrato_4": 5.834299709915722,
    "FAMI_ESTRATOVIVIENDA_Estrato_5": 12.284528681022799,
    "FAMI_ESTRATOVIVIENDA_Estrato_6": 14.402787353688034
}

# Intercepto del modelo para el Saber 11
intercepto_saber11 = 270.8199229465451

# Varianza residual del modelo para el Saber 11 (ejemplo hipotético)
varianza_residual = 100
# Crear una lista para almacenar los divs generados dinámicamente
divs = []
# Iterar sobre los valores de la serie datos_description
for nombre_coeficiente, valor_coeficiente in coeficientes_ordenados:
    # Calcular la intensidad del color azul para el coeficiente
    color_azul = 400 - int((valor_coeficiente - min(coeficientes.values())) / (max(coeficientes.values()) - min(coeficientes.values())) * 350)

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
def predecir_saber11(valores, coeficientes, intercepto, varianza_residual):
    y = intercepto
    for clave, valor in valores.items():
        if clave in coeficientes:
            y += coeficientes[clave] * valor
    error_estandar = np.sqrt(varianza_residual)
    grados_libertad = len(valores) - 1  # Número de coeficientes menos el intercepto
    valor_critico = t.ppf(0.975, df=grados_libertad)  # Nivel de confianza del 95%
    intervalo = valor_critico * error_estandar
    return y, intervalo, grados_libertad

# Función para cargar el archivo CSV y mostrar la predicción del Saber 11 y el intervalo de confianza
# Luego, modifica la función parse_contents para corregir la creación de la gráfica de campana
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Realizar la predicción utilizando los datos del archivo
    valores = {columna: df[columna].iloc[0] for columna in df.columns}
    prediccion_saber11, intervalo, grados_libertad = predecir_saber11(valores, coeficientes_saber11, intercepto_saber11, varianza_residual)
    
    # Crear una distribución normal para el puntaje predicho y el intervalo de confianza
    x_values = np.linspace(prediccion_saber11 - 3 * intervalo, prediccion_saber11 + 3 * intervalo, 100)
    y_values = t.pdf((x_values - prediccion_saber11) / intervalo, df=grados_libertad) / intervalo
    
    # Crear la figura de la gráfica de campana
    bell_curve = go.Figure()
    bell_curve.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Curva de Campana'))
    bell_curve.update_layout(title='Predicción del Saber 11 y Intervalo de Confianza',
                             xaxis_title='Puntaje predicho del Saber 11',
                             yaxis_title='Densidad de probabilidad',
                             plot_bgcolor='rgba(0,0,0,0)', # Establece el color de fondo del gráfico como transparente
                             paper_bgcolor='white' # Establece el color de fondo del área de la gráfica como blanco
                             )
    
    return dcc.Graph(figure=bell_curve)

# Definir el color azul personalizado para el fondo y las gráficas
color_azul_personalizado = 'white'

# Inicializar la aplicación de Dash
app = dash.Dash(__name__)
# LAYOUT
app.layout = html.Div([
    html.Div(
        style={'backgroundColor': color_azul_personalizado},
        children=[
            html.H1("Análisis de Datos - Saber 11", style={'textAlign': 'center', 'color': 'black'}),
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
  app.run_server(debug=True)