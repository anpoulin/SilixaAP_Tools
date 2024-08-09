import matplotlib.pyplot as plt

# Data
days_of_year = [292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304]
triggers = [202, 219, 147, 349, 314, 1243, 788, 1166, 1900, 1900, 603, 153, 529]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(days_of_year, triggers, color='skyblue')

# Add labels and title
plt.xlabel('Day of Year (DOY)')
plt.ylabel('Number of Triggers')
plt.title('Number of Triggers per Day of Year')

# Show plot
plt.show()