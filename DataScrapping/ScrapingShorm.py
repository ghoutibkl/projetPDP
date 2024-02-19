from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
from datetime import datetime, timedelta

# URL de base avec le format de date
base_url = "https://maree.shom.fr/harbor/PAUILLAC/wl/0?date={}&utc=standard"

# Utilise un navigateur Chrome (tu dois avoir chromedriver installé et dans le PATH)
driver = webdriver.Chrome()

# Crée un fichier CSV pour stocker les données
with open('donnees_marees_2020_2023.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Écriture de l'en-tête du fichier CSV
    csvwriter.writerow(['Date', 'Heure', 'Hauteur'])

    # Itération sur les jours de l'année 2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    current_date = start_date

    while current_date <= end_date:
        # Formatage de la date dans le format attendu par le site
        date_str = current_date.strftime("%Y-%m-%d")
        url = base_url.format(date_str)

        # Ouvre la page dans le navigateur
        driver.get(url)

        # Attends que la page soit complètement chargée (tu peux ajuster ce délai si nécessaire)
        driver.implicitly_wait(10)

        # Trouve la table par son sélecteur CSS
        table = driver.find_element(By.CSS_SELECTOR, 'table')

        # Parcours les lignes de la table (en commençant par la deuxième ligne pour éviter les en-têtes répétées)
        for row in table.find_elements(By.TAG_NAME, 'tr')[1:]:
            # Extrait les cellules de chaque ligne
            cells = row.find_elements(By.XPATH, './/th | .//td')
            # Écrit les données dans le fichier CSV
            csvwriter.writerow([date_str] + [cell.text.strip() for cell in cells[1:]])

        # Passe à la prochaine date
        current_date += timedelta(days=1)

# Ferme le navigateur
driver.quit()

print("Les données ont été extraites avec succès et enregistrées dans 'donnees_marees_2023.csv'.")
