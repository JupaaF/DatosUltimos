import matplotlib.pyplot as plt
import pandas as pd

def main():

    #Leer ambos archivos CSV en mismo DataFrame
    horizontal_df = pd.read_csv('dataset/processed/stable_packings_horizontal.csv')
    vertical_df = pd.read_csv('dataset/processed/stable_packings_vertical.csv')
    combined_df = pd.concat([horizontal_df, vertical_df], ignore_index=True)

    #Crear histogramas para cada columna
    for column in combined_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(combined_df[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histograma de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'histogram_{column}.png')  # Guardar el histograma como imagen
        plt.close()  # Cerrar la figura para liberar memoria


if __name__ == "__main__":
    main()
