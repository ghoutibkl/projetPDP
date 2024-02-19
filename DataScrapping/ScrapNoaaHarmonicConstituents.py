from selenium import webdriver
from bs4 import BeautifulSoup
import csv

# Specify the path to your web driver


# URL of the website to scrape
url = "https://tidesandcurrents.noaa.gov/harcon.html?id=9447130"

# Create a webdriver instance
driver = webdriver.Chrome()

# Load the webpage
driver.get(url)

# Wait for the page to load completely (you may need to adjust the sleep duration)
driver.implicitly_wait(10)

# Get the page source after it has been fully loaded
html = driver.page_source

# Close the webdriver
driver.quit()

# Parse the HTML content of the page
soup = BeautifulSoup(html, 'html.parser')

# Find the table containing the data
table = soup.find('table', {'class': 'table table-striped'})

# Initialize data list to store scraped data
data = []

# Extract table rows
rows = table.find_all('tr')

# Extract table headers
headers = [header.text.strip() for header in rows[0].find_all('th')]

# Iterate over rows skipping the header row
for row in rows[1:]:
    # Extract table data
    row_data = [data.text.strip() for data in row.find_all('td')]
    # Append data to the list
    data.append(row_data)

# Write data to CSV file
with open('harmonic_constituents.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write headers
    writer.writerow(headers)
    # Write rows
    writer.writerows(data)

print("Data has been scraped and saved to harmonic_constituents.csv")
