import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# View the first few rows
print(df.head())

# Basic info
print(df.info())

# Calculate average age (or any numeric column)
average_age = df['Age'].mean()
print(f"Average Age: {average_age}")

# --------------------------
# Bar Chart: Count by Pclass
# --------------------------
pclass_counts = df['Pclass'].value_counts()

plt.figure(figsize=(8,6))
pclass_counts.plot(kind='bar', color='skyblue')
plt.title('Passenger Count by Class (Pclass)')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# --------------------------
# Scatter Plot: Age vs Fare
# --------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['Fare'], alpha=0.6, color='green')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# --------------------------
# Heatmap: Correlation Matrix
# --------------------------
corr = df[['Age', 'Fare', 'Pclass']].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
