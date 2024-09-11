import matplotlib.pyplot as plt
import datetime

# Data
days_of_year = [
    274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
    293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
    312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
    350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365
]
triggers = [
    13, 0, 113, 77, 223, 1030, 1174, 919, 26, 0, 529, 0, 0, 0, 0, 0, 0, 0, 202, 219, 147, 349, 314,
    1243, 788, 1166, 1900, 1900, 603, 153, 529, 656, 1117, 589, 1499, 814, 4420, 2147, 1589, 2109,
    1760, 1942, 2463, 2071, 1849, 2131, 1862, 2091, 2048, 1592, 1781, 1329, 2014, 1852, 1457, 3152,
    3229, 1179, 1750, 1145, 1950, 1901, 1707, 1502, 2500, 3167, 1443, 1101, 841, 1238, 427, 437,
    371, 227, 267, 565, 522, 1509, 1109, 2302, 1088, 2038, 817, 1158, 2787, 1538, 3666, 466, 391,
    135, 113, 513
]

# Convert DOY to dates
year = 2022
dates = [datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1) for doy in days_of_year]
formatted_dates = [date.strftime('%m/%d') for date in dates]

# Create bar chart
plt.figure(figsize=(18, 8))
plt.bar(formatted_dates, triggers, color='skyblue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Number of Triggers')
plt.title('Number of Triggers per 24 Hours')

# Rotate date labels for better readability
plt.xticks(rotation=90)

# Show plot
plt.show()
