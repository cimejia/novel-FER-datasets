# Distributing facial images into gender, age and race subcategories
import csv
import shutil

path = '/mnt/datosnas/Users/cmejia/EMOTION-DETECTION/NHFI-NUEVO-AGR/6/' # For each dataset and emotion category
with open('SURPRISE-results.csv','r') as file:
    reader = csv.reader(file, delimiter = ',')
    for row in reader:
        img = row[0]
        age = row[1]
        race = row[2]
        gender = row[3]
        source = path + img
        target = path + gender + "-" + age + "-" + race + "/" + img
        print("Moving " + source + " to " + target)
        shutil.move(source, target)
