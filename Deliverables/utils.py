# --- Standard Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import zipfile

# --- Evaluation Metrics ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def check_code_description_combinations(df, code_column, description_column):
    # Count unique combinations of Code and Description
    unique_combinations = df[[code_column, description_column]].drop_duplicates()
    print(f"Number of unique combinations of {code_column} and {description_column}: {unique_combinations.shape[0]}")

    # Count total unique codes and descriptions individually
    code_count_sum = df[code_column].nunique()
    description_count_sum = df[description_column].nunique()

    print(f"Total unique {code_column} values: {code_count_sum}")
    print(f"Total unique {description_column} values: {description_count_sum}")

    # Check if the unique combination count matches the individual totals
    if code_count_sum == description_count_sum == unique_combinations.shape[0]:
        print(f"The number of unique combinations matches the total counts of {code_column} and {description_column}.")
    else:
        print(f"There is a discrepancy between the number of unique combinations and the total counts of {code_column} and {description_column}.")
        
        
        
        
        
##3.3.2. Multivariate Analysis    
def plot_count_for_binary_and_categorical(data, binary_vars, categorical_vars):
    for binary_var in binary_vars:
        print(f"Binary Variable: {binary_var}")
        for cat_var in categorical_vars:
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(data=data, x=cat_var, hue=binary_var, palette='viridis')
            plt.title(f'Distribution of {binary_var} per {cat_var}')
            plt.xticks(rotation=90, ha='right')  
            
            # Add counts above bars
            for p in ax.patches:
                ax.annotate(f'{p.get_height()}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')
                
            plt.tight_layout() 
            plt.show()
            
            
### 3.2.10. Handling Zip code format
# Define a function to standardize ZIP codes by adding leading zeros if necessary
def convert_zip_code(zip_code):
    if pd.isna(zip_code):
        return np.nan
    zip_code_str = str(zip_code).split('.')[0]  # Convert to string and remove decimal if float
    if len(zip_code_str) == 4:
        return zip_code_str.zfill(5)  # Add leading zero if length is 4
    return zip_code_str


#### 5.2. Missing Values
# Define a function to fill missing values based on 'Industry Code' using training means

# Define a function to impute 'Industry Code' based on 'Carrier Name' and 'Carrier Type' using training mode
def impute_industry_code(row):
    if pd.isna(row['Industry Code']):
        # Get the mode of the industry code for the given carrier name and carrier type from training set
        return industry_code_mode_train.get((row['Carrier Name'], row['Carrier Type']), row['Industry Code'])
    return row['Industry Code']

def impute_average_weekly_wage(row):
    if pd.isna(row['Average Weekly Wage']):
        return industry_means_train.get(row['Industry Code'], row['Average Weekly Wage'])
    return row['Average Weekly Wage']


# Define a function to impute 'Zip Code' based on 'County of Injury' and 'District Name' using training mode
def impute_zip_code(row):
    if pd.isna(row['Zip Code']):
        # Get the mode of the zip code for the given county and district from training set
        return zip_code_mode_train.get((row['County of Injury'], row['District Name']), row['Zip Code'])
    return row['Zip Code']


# Define a function to impute 'Birth Year' based on 'Assembly Date' and 'Age at Injury'
def impute_birth_year(row):
    if pd.isna(row['Birth Year']):
        if pd.notna(row['Assembly Date']) and pd.notna(row['Age at Injury']):
            # Calculate birth year by subtracting age at injury from assembly year
            assembly_year = row['Assembly Date'].year
            return float(assembly_year - row['Age at Injury'])
    return row['Birth Year']


### 6.3. Days to First Hearing
# Define a function to calculate the number of days between 'Accident Date' and 'First Hearing Date'
def calculate_hearing_days(row):
    if pd.notna(row['First Hearing Date']):
        return (row['First Hearing Date'] - row['Accident Date']).days
    return 0  # If no hearing date exists, represent as 0


#### 6.8. Promptness category
def categorize_promptness(df, date1_col, date2_col, new_col_name):
    """
    Calculate and categorize promptness between two date columns.

    Parameters:
    - df: The DataFrame to process.
    - date1_col: The column representing the first date (e.g., Assembly Date).
    - date2_col: The column representing the second date (e.g., Accident Date).
    - new_col_name: The name of the new categorical column for promptness.

    Returns:
    - Updated DataFrame with new categorized promptness column.
    """
    # Calculate promptness in days and categorize it
    df[new_col_name] = pd.cut(
        (df[date1_col] - df[date2_col]).dt.days,
        bins=[0, 7, 14, 30, 90, 180, 365, float('inf')],
        labels=['Until 1 weeks', 'Between 1 and 2 weeks', 'Between 2 weeks and 1 month', 
                '1 to 3 months', '3 to 6 months', '6 months to 1 year', 'More than 1 year'],
        right=True
    )
    return df

#Function used in Modeling - File: Deliverables/Notebooks 2., 3., 4. and 5.
# Define a function to print metrics and plot a colorful confusion matrix
def metrics(y_train, pred_train, y_val, pred_val):
    # Print classification report for training data
    print('___________________________________________________________________________________________________________')
    print('                                                     TRAIN                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_train, pred_train))
    # train_cm = confusion_matrix(y_train, pred_train)
    # print(train_cm)
    
    # Plot confusion matrix for training data
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title('Training Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Print classification report for validation data
    print('___________________________________________________________________________________________________________')
    print('                                                VALIDATION                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_val, pred_val))
    # val_cm = confusion_matrix(y_val, pred_val)
    # print(val_cm)

    # Plot confusion matrix for validation data
    plt.subplot(1, 2, 2)
    sns.heatmap(val_cm, annot=True, cmap='Oranges', fmt='d', cbar=False)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Show the plots
    plt.tight_layout()
    plt.show()
