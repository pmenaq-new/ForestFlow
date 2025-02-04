import numpy as np
import pandas as pd
import numpy_financial as nf
from scipy.stats import norm, triang, uniform, weibull_min, lognorm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- Definición de la estructura de columnas y sus tipos ---
VARIABLES_FLUJO = {
    'Variable_ID': str,           # Identificador único
    'Nombre': str,                # Descripción de la variable
    'Tipo_Variable': str,         # "Costo Fijo", "Costo Variable", "Cantidad" o "Precio"
    'Unidad': str,                # Unidad de medida (ej. CLP, USD, m³, etc.)
    'Valor_Base': float,          # Valor central esperado
    'Valor_Mín': float,           # Valor mínimo estimado
    'Valor_Máx': float,           # Valor máximo estimado
    'Desviación_Estandar': float, # Desviación estándar para simulación
    'Distribución': str,          # Tipo de distribución ("normal", "triangular", "uniforme", "weibull", "log-normal")
    'Frecuencia': str,            # "único" o "anual"
    'Inicio': int,                # Período en que inicia la variable
    'Fin': int,                   # Período en que finaliza la aplicación
    'Dependencia': str,           # Ej.: "Ninguna", "Volumen", "Divisa", etc.
    'Observaciones': str         # Comentarios opcionales
}

def validar_dataframe(df, variables=VARIABLES_FLUJO):
    """
    Verifica que el DataFrame contenga todas las columnas requeridas y
    convierte cada columna al tipo indicado en el diccionario.
    """
    missing = [col for col in variables.keys() if col not in df.columns]
    if missing:
        raise ValueError("Faltan columnas en el archivo: " + ", ".join(missing))
    for col, dtype in variables.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"Error convirtiendo la columna {col} a {dtype}: {e}")
    return df

# --- Clase para la simulación del flujo de caja basado en parámetros de Excel ---
class SimulacionFlujoCajaExcel:
    def __init__(self, excel_file, num_iteraciones, duracion, tasa_descuento=0.05):
        """
        Parámetros:
          - excel_file: ruta al archivo Excel con la parametrización.
          - num_iteraciones: número de simulaciones (Monte Carlo).
          - duracion: cantidad de períodos (ej. años) del proyecto.
          - tasa_descuento: tasa para el cálculo del VAN.
        """
        self.excel_file = excel_file
        self.num_iteraciones = num_iteraciones
        self.duracion = duracion
        self.tasa_descuento = tasa_descuento
        # Cargar y validar el archivo Excel
        self.df_variables = validar_dataframe(pd.read_excel(excel_file))
        # Separar las variables según su tipo
        self.costos_fijos = self.df_variables[self.df_variables['Tipo_Variable'].str.lower() == "costo fijo"]
        self.costos_variables = self.df_variables[self.df_variables['Tipo_Variable'].str.lower() == "costo variable"]
        self.cantidades = self.df_variables[self.df_variables['Tipo_Variable'].str.lower() == "cantidad"]
        self.precios = self.df_variables[self.df_variables['Tipo_Variable'].str.lower() == "precio"]
        # Agrupar ingresos provenientes de Precio y Cantidad
        self.revenue_groups = self._agrupar_revenue()

    def _simular_variable(self, row):
        """
        Simula un valor para la variable según la distribución y parámetros.
        """
        distribucion = row['Distribución'].lower()
        valor_base = row['Valor_Base']
        desv = row['Desviación_Estandar']
        vmin = row['Valor_Mín']
        vmax = row['Valor_Máx']
        
        if distribucion == "normal":
            return max(norm.rvs(loc=valor_base, scale=desv), 0)
        elif distribucion == "triangular":
            c = 0.5 if (vmax - vmin) == 0 else (valor_base - vmin) / (vmax - vmin)
            return triang.rvs(c, loc=vmin, scale=vmax - vmin)
        elif distribucion in ["uniforme", "uniform"]:
            return uniform.rvs(loc=vmin, scale=vmax - vmin)
        elif distribucion == "weibull":
            return weibull_min.rvs(1.5, scale=valor_base)
        elif distribucion in ["log-normal", "lognormal"]:
            sigma = np.sqrt(np.log(1 + (desv/valor_base)**2)) if valor_base != 0 else 1
            mu = np.log(valor_base) - 0.5 * sigma**2 if valor_base != 0 else 0
            return lognorm.rvs(s=sigma, scale=np.exp(mu))
        else:
            return valor_base

    def _agrupar_revenue(self):
        """
        Agrupa las variables de ingresos que provienen de la combinación de Precio y Cantidad.
        Se asume la convención en la columna 'Nombre':
          - Ejemplo: "ProductoA_Precio" y "ProductoA_Cantidad"
        """
        revenue_groups = {}
        # Procesar precios y buscar la cantidad asociada
        for _, precio_row in self.precios.iterrows():
            nombre = precio_row['Nombre']
            if "_Precio" in nombre:
                prefix = nombre.split("_Precio")[0]
            else:
                prefix = nombre
            match = self.cantidades[self.cantidades['Nombre'] == f"{prefix}_Cantidad"]
            if not match.empty:
                revenue_groups[prefix] = {
                    'precio': precio_row,
                    'cantidad': match.iloc[0]
                }
            else:
                revenue_groups[nombre] = {
                    'precio': precio_row,
                    'cantidad': None
                }
        # Incluir cantidades sin par
        for _, cant_row in self.cantidades.iterrows():
            nombre = cant_row['Nombre']
            if "_Cantidad" in nombre:
                prefix = nombre.split("_Cantidad")[0]
            else:
                prefix = nombre
            if prefix not in revenue_groups:
                revenue_groups[prefix] = {
                    'precio': None,
                    'cantidad': cant_row
                }
        return revenue_groups

    def simular(self):
        """
        Realiza la simulación Monte Carlo. Para cada iteración se genera un vector de flujo
        de caja de longitud 'duracion' sumando las contribuciones de cada variable y se calcula VAN y TIR.
        Retorna una lista de diccionarios (una por iteración).
        """
        resultados = []
        for iteracion in range(self.num_iteraciones):
            flujo_periodos = np.zeros(self.duracion)
            # Simulación para cada variable definida en el Excel
            for _, row in self.df_variables.iterrows():
                inicio = int(row['Inicio'])
                fin = int(row['Fin'])
                frecuencia = row['Frecuencia'].lower()
                if frecuencia in ["único", "unico"]:
                    periodos_activos = [inicio]
                elif frecuencia == "anual":
                    periodos_activos = list(range(inicio, min(fin + 1, self.duracion)))
                else:
                    periodos_activos = [inicio]
                valor_sim = self._simular_variable(row)
                tipo = row['Tipo_Variable'].lower()
                # Se asume que los costos (fijo/variable) son negativos y otros positivos.
                contrib = -valor_sim if tipo in ["costo fijo", "costo variable"] else valor_sim
                for t in periodos_activos:
                    flujo_periodos[t] += contrib

            # Procesar ingresos agrupados (Precio x Cantidad)
            for grupo, componentes in self.revenue_groups.items():
                precio_row = componentes.get('precio')
                cantidad_row = componentes.get('cantidad')
                precio_val = self._simular_variable(precio_row) if precio_row is not None else None
                cantidad_val = self._simular_variable(cantidad_row) if cantidad_row is not None else None
                if (precio_val is not None) and (cantidad_val is not None):
                    ingreso = precio_val * cantidad_val
                elif precio_val is not None:
                    ingreso = precio_val
                elif cantidad_val is not None:
                    ingreso = cantidad_val
                else:
                    ingreso = 0
                # Usar la información de precio o cantidad para definir el período de aplicación
                if precio_row is not None:
                    inicio = int(precio_row['Inicio'])
                    fin = int(precio_row['Fin'])
                    frecuencia = precio_row['Frecuencia'].lower()
                elif cantidad_row is not None:
                    inicio = int(cantidad_row['Inicio'])
                    fin = int(cantidad_row['Fin'])
                    frecuencia = cantidad_row['Frecuencia'].lower()
                else:
                    inicio, fin, frecuencia = 0, 0, "único"
                if frecuencia in ["único", "unico"]:
                    periodos_activos = [inicio]
                elif frecuencia == "anual":
                    periodos_activos = list(range(inicio, min(fin + 1, self.duracion)))
                else:
                    periodos_activos = [inicio]
                for t in periodos_activos:
                    flujo_periodos[t] += ingreso

            van = nf.npv(self.tasa_descuento, flujo_periodos)
            tir = nf.irr(flujo_periodos)
            resultados.append({
                "Iteracion": iteracion,
                "Flujo": flujo_periodos,
                "VAN": van,
                "TIR": tir
            })
        return resultados

    def resumen_simulacion(self):
        """
        Ejecuta la simulación y retorna un DataFrame resumen con VAN y TIR por iteración.
        """
        resultados = self.simular()
        resumen = pd.DataFrame({
            "Iteracion": [r["Iteracion"] for r in resultados],
            "VAN": [r["VAN"] for r in resultados],
            "TIR": [r["TIR"] for r in resultados]
        })
        return resumen

    def graficar_histogramas(self):
        """
        Grafica histogramas de la distribución del VAN y la TIR a partir de la simulación.
        """
        resumen = self.resumen_simulacion()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(resumen['VAN'], kde=True)
        plt.title("Distribución del VAN")
        plt.subplot(1, 2, 2)
        sns.histplot(resumen['TIR'], kde=True)
        plt.title("Distribución del TIR")
        plt.tight_layout()
        plt.show()

# --- Función para sensibilidad en duración y tasa de descuento ---
def simular_sensibilidad_proyecto(excel_file, num_iteraciones, duraciones, discount_rates, output_csv):
    """
    Itera sobre combinaciones de duración del proyecto y tasa de descuento.
    
    Parámetros:
      - excel_file: ruta al archivo Excel con los parámetros.
      - num_iteraciones: número de simulaciones por combinación.
      - duraciones: lista de duraciones (por ejemplo, [15, 20, 25]).
      - discount_rates: lista de tasas de descuento (por ejemplo, [0.05, 0.1, 0.15]).
      - output_csv: ruta del archivo CSV de salida.
    
    Retorna un DataFrame combinado con el resumen de cada iteración, agregando columnas para
    la duración y la tasa de descuento.
    """
    resultados = []
    for duracion in duraciones:
        for tasa in discount_rates:
            sim = SimulacionFlujoCajaExcel(excel_file, num_iteraciones, duracion, tasa)
            resumen = sim.resumen_simulacion()
            resumen["Duracion"] = duracion
            resumen["Tasa_Descuento"] = tasa
            resultados.append(resumen)
    df_resultados = pd.concat(resultados, ignore_index=True)
    df_resultados.to_csv(output_csv, index=False)
    return df_resultados

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Se asume que el archivo "parametros_flujo.xlsx" tiene la estructura definida.
    ruta_excel = "data\parametros_flujo_prueba.xlsx"
    num_iteraciones = 1000
    
    # Listas de sensibilidad para duración y tasa de descuento
    lista_duraciones = [15, 20, 25]          # Por ejemplo, proyectos de 15, 20 y 25 años
    lista_tasas = [0.05, 0.1, 0.15]           # Tasas de descuento del 5%, 10% y 15%
    
    archivo_salida = "sensibilidad_resultados.csv"
    df_sensibilidad = simular_sensibilidad_proyecto(ruta_excel, num_iteraciones,
                                                    lista_duraciones, lista_tasas,
                                                    archivo_salida)
    print("Resumen de sensibilidad:")
    print(df_sensibilidad.head())
    
    # Para visualizar los histogramas de una simulación particular, se puede instanciar:
    simulacion = SimulacionFlujoCajaExcel(ruta_excel, num_iteraciones, duracion=20, tasa_descuento=0.05)
    simulacion.graficar_histogramas()
