import json 
import matplotlib.pyplot as plt

filename = 'c:\Projects\TOYO\P40_PTW1.json'

# Open the JSON file
with open(filename, 'r') as file:
    data = json.load(file)

first_table = data["confDepths"]["firstTable"]

FD = [row[0] for row in first_table]
MD = [row[1] for row in first_table]

plt.title(filename)
plt.xlabel('FD (m)')
plt.ylabel('MD (m)')
plt.plot(FD,MD)


plt.show()


print(data)