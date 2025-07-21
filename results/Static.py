import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the CSV file
# df = pd.read_csv('/mnt/disk1/SonDinh/SonDinh/AES_project/tools/grammar/FullData_3part_train.csv')
# de_test = pd.read_csv('/mnt/disk1/SonDinh/SonDinh/AES_project/tools/grammar/FullData_3part_test.csv')
# # Combine the two dataframes
# df = pd.concat([df, de_test], ignore_index=True)
df = pd.read_csv('/media/gpus/Data/AES/ESL-Grading/data/Full/Full_train_aug.csv')
# Calculhttps://drive.google.com/drive/u/0/folders/1kBfpH1SEyku-B3QZ5gJEl7f7UeuXvXILate grammar of unique grammar scores
grammar_freq = df['grammar'].value_counts().sort_index()
grammar_percent = (grammar_freq / len(df) * 100).round(2)

# Create summary dataframe
grammar_summary = pd.DataFrame({
    'grammar': grammar_freq,
    'Percentage': grammar_percent
})

# Prepare the plot
plt.figure(figsize=(16, 10))

# Subplot 1: Bar Plot of grammar Score Frequencies
plt.subplot(1, 2, 1)
grammar_summary['grammar'].plot(kind='bar', color='skyblue', edgecolor='navy')
plt.title('grammar of grammar Scores', fontsize=15)
plt.xlabel('grammar Score', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Subplot 2: Pie Chart of grammar Score Percentages
plt.subplot(1, 2, 2)
plt.pie(grammar_summary['Percentage'], 
        labels=[f'{index}: {percent}%' for index, percent in zip(grammar_summary.index, grammar_summary['Percentage'])],
        autopct='%1.1f%%',
        colors=plt.cm.Pastel1.colors)
plt.title('Distribution of grammar Scores', fontsize=15)

plt.tight_layout()

# Save the plot
plt.savefig('grammar_score_distribution_part3.png', dpi=300, bbox_inches='tight')

# Print summary statistics
print("grammar Score grammar Analysis:")
print(grammar_summary)

# Additional statistical insights
print("\ngrammar Score Statistical Summary:")
print(df['grammar'].describe())

# Save summary to CSV
grammar_summary.to_csv('grammar_score_grammar_part3.csv')