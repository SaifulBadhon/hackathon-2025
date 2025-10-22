import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
# Load validation results (CSV with 'true' and 'pred' columns)
df = pd.read_csv("E:/Hackathon UNT 2025/PD HAR/model_vs_llm_predictions.csv")
print(df)
# Compute confusion matrix
cm = confusion_matrix(df["true_label"], df["llm_prediction"])
print(classification_report(df["true_label"], df["model_prediction"]))
labels = sorted(df["true"].unique())

# Plot confusion matrix with larger fonts
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            annot_kws={"fontsize": 14})  # font size for numbers

plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)
plt.title("Confusion Matrix", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
main_path = 'E:/Hackathon UNT 2025/pads-parkinsons-disease-smartwatch-dataset-1.0.0/pads-parkinsons-disease-smartwatch-dataset-1.0.0/'
path = 'all_subjects.csv'
meta_path = 'preprocessed/file_list.csv'
df = pd.read_csv(main_path+meta_path)
print(df)
Convert boolean-like columns to string (optional)
df["appearance_in_kinship"] = df["appearance_in_kinship"].astype(str)
df["appearance_in_first_grade_kinship"] = df["appearance_in_first_grade_kinship"].astype(str)

# -----------------------------
# Add BMI column
# -----------------------------
df["BMI"] = df["weight"] / (df["height"] / 100) ** 2

# -----------------------------
# Summary Statistics
# -----------------------------
print("Missing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nUnique Values:\n", df.nunique())
print("\nStatistical Summary:\n", df.describe())
df["condition_group"] = df["condition"].apply(
    lambda x: "Parkinson's" if x == "Parkinson's"
    else "Healthy" if x == "Healthy"
    else "Other"
)

# Count the number of patients in each group
grouped_counts = df["condition_group"].value_counts().reset_index()
grouped_counts.columns = ["Condition Group", "Count"]

# Plot the grouped condition distribution
plt.figure(figsize=(8, 6))
plt.bar(grouped_counts["Condition Group"], grouped_counts["Count"])
plt.title("Grouped Condition Distribution: Healthy vs Parkinson's vs Other", fontsize=16)
plt.xlabel("Condition Group", fontsize=14)
plt.ylabel("Number of Subjects", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()
# -----------------------------
# Plot numeric distributions
# -----------------------------
numeric_cols = ['age', 'age_at_diagnosis', 'height', 'weight', 'BMI']
for col in numeric_cols:
    plt.figure(figsize=(10,8))
    if col == 'age_at_diagnosis':
         plt.xlabel('Age at diagnosis',fontsize=20)
    else:
        plt.xlabel(col,fontsize=20)
    # print(df[(df['condition']!="Healthy") and (df[col]==0]))
    sns.histplot(df[df['condition']!="Healthy"][col], kde=True)
    # plt.title(f"Distribution of {col}")

    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Categorical Feature Counts
# -----------------------------
categorical_cols = [
    'condition', 'gender', 'handedness', 'appearance_in_kinship',
    'appearance_in_first_grade_kinship', 'effect_of_alcohol_on_tremor', 'label'
]

for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())

# -----------------------------
# Correlation Matrix
# -----------------------------
correlation = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
