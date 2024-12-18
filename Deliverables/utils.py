# --- Standard Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import zipfile
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif



# --- Evaluation Metrics ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

############################################################################################################
#----------------------------------- Functions used in Notebooks 1. ---------------------------------------
############################################################################################################

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

# Binary Variables against Categorical Variables
def plot_count_for_binary_and_categorical(data, binary_vars, categorical_vars):
    # Loop through each binary variable in the list
    for binary_var in binary_vars:
        print(f"Binary Variable: {binary_var}\n")  # Print the binary variable being plotted
        # Loop through each categorical variable in the list
        for categorical_var in categorical_vars:
            plt.figure(figsize=(12, 6))  # Adjust figure size
            ax = sns.countplot(data=data, x=categorical_var, hue=binary_var, palette="deep")  # Generate count plot
            
            # Add annotations to display counts on top of the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=3)
            
            plt.title(f"Distribution of {binary_var} per {categorical_var}")  # Set the title
            plt.xlabel(categorical_var)  # Set x-axis label
            plt.ylabel("Count")  # Set y-axis label
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.legend(title=binary_var, loc='upper right')  # Set legend
            plt.show()  # Display the plot

# Discrete Variables against Binary Variables     
def plot_count_for_binary_and_discrete(data, binary_vars, discrete_vars):
    # Loop through each binary variable
    for binary_var in binary_vars:
        print(f"Binary Variable: {binary_var}\n")  # Print the binary variable being plotted
        
        # Loop through each discrete variable
        for discrete_var in discrete_vars:
            plt.figure(figsize=(16, 8))  # Increase figure size for better clarity
            
            # Create the count plot with improved aesthetics
            ax = sns.countplot(
                data=data, 
                x=discrete_var, 
                hue=binary_var, 
                palette="muted", 
                linewidth=0.5, 
                edgecolor="gray"
            )
            
            # Add annotations to display counts on top of the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=3)
            
            # Improve titles and labels for clarity
            plt.title(f"Distribution of {binary_var} by {discrete_var}", fontsize=14, fontweight='bold')  
            plt.xlabel(discrete_var, fontsize=12)  
            plt.ylabel("Count", fontsize=12)  
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            
            # Optimize layout to avoid overlapping elements
            plt.tight_layout()  
            plt.legend(title=binary_var, loc='upper right')  # Add legend
            
            plt.show()  # Display the plot

# Discrete Variables against Categorical Variables    
def plot_count_for_discrete_and_categorical(data, discrete_vars, categorical_vars):
    # Loop through each binary variable
    for discrete_var in discrete_vars:
        print(f"Discrete Variable: {discrete_var}\n")  # Print the binary variable being plotted
        
        # Loop through each categorical variable
        for categorical_var in categorical_vars:
            plt.figure(figsize=(16, 8))  # Increase figure size for better clarity
            
            # Create the count plot with aesthetics for categorical variables
            ax = sns.countplot(
                data=data, 
                x=categorical_var, 
                hue=discrete_var, 
                palette="muted", 
                order=data[categorical_var].value_counts().index,  # Order categories by frequency
                linewidth=0.5, 
                edgecolor="gray"
            )
            
            # Add annotations to display counts on top of the bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=3)
            
            # Improve titles and labels for clarity
            plt.title(f"Distribution of {discrete_var} by {categorical_var}", fontsize=14, fontweight='bold')  
            plt.xlabel(categorical_var, fontsize=12)  
            plt.ylabel("Count", fontsize=12)  
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            
            # Optimize layout to avoid overlapping elements
            plt.tight_layout()  
            plt.legend(title=discrete_var, loc='upper right')  # Add legend
            
            plt.show()  # Display the plot

          
            
### 3.2.10. Handling Zip code format
# Define a function to standardize ZIP codes by adding leading zeros if necessary
def convert_zip_code(zip_code):
    if pd.isna(zip_code):
        return np.nan
    zip_code_str = str(zip_code).split('.')[0]  # Convert to string and remove decimal if float
    if len(zip_code_str) == 4:
        return zip_code_str.zfill(5)  # Add leading zero if length is 4
    return zip_code_str

############################################################################################################
#----------------------------------- Functions used in Notebooks 2. ---------------------------------------
############################################################################################################

#### 4.1. Outliers
def analyze_numerical_outliers(df, columns):
    """
    Analyze numerical variables for potential outliers using the Interquartile Range (IQR) method.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame containing numerical variables.
    columns : list of str
        The list of column names to analyze for outliers.

    Returns:
    numerical_outliers : dict
        A dictionary where the keys are column names and the values are the count of outliers detected 
        in each column.

    Method:
    - For each column, calculate the IQR (Interquartile Range).
    - Define the lower and upper bounds for outliers as:
        - Lower Bound: Q1 - 1.5 * IQR
        - Upper Bound: Q3 + 1.5 * IQR
    - Identify rows with values outside these bounds and count them.
    """
    numerical_outliers = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)  # First quartile (25%)
        Q3 = df[column].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1                   # Interquartile range
        lower_bound = Q1 - 1.5 * IQR    # Lower bound for outliers
        upper_bound = Q3 + 1.5 * IQR    # Upper bound for outliers
        # Identify outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_count = outliers.shape[0]
        numerical_outliers[column] = outliers_count
    return numerical_outliers



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




############################################################################################################
#-------------- Functions used in Modeling - File: Deliverables/Notebooks 3., 4., 5. and 6. ------------
############################################################################################################

### In feature selection ####
# Used to plot correlation matrix
def cor_heatmap(cor, name):
    plt.figure(figsize=(18,12))
    sns.heatmap(data = cor.round(2), annot = True, cmap = 'viridis', linecolor = 'white', linewidth=0.5, fmt='.2', mask=np.triu(cor, k=0))
    plt.title(f'{name} Correlation Matrix', fontdict = {'fontsize': 20})
    plt.show()
    
#Used to plot lasso coefficients 
def plot_importance(coef, name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(3,5))
    imp_coef.plot(kind="barh", color="c")
    plt.title("Feature importance using " + name + " Model")
    plt.show()

#This function is it to find the optimal number of features using Recursive Feature Elimination (RFE) and evaluates using F1 Macro score.
def find_optimal_features_with_rfe(model, X_train, y_train, X_val, y_val, max_features=8):
    """
    Finds the optimal number of features using Recursive Feature Elimination (RFE) 
    and evaluates using F1 Macro score.
    
    Parameters:
    - model: The machine learning model (e.g., LogisticRegression()).
    - X_train: Scaled training feature set (numpy array or DataFrame).
    - y_train: Encoded training target labels.
    - X_val: Scaled validation feature set (numpy array or DataFrame).
    - y_val: Encoded validation target labels.
    - max_features: Maximum number of features to evaluate (default=8).

    Returns:
    - best_features: Optimal number of features for the highest F1 Macro score.
    - best_score: The highest F1 Macro score achieved.
    - scores_list: List of F1 Macro scores for each number of features.
    """
    nof_list = np.arange(1, max_features + 1)
    high_score = 0
    best_features = 0
    scores_list = []

    for n in nof_list:
        rfe = RFE(model, n_features_to_select=n)
        
        # Transform training and validation sets with RFE
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_val_rfe = rfe.transform(X_val)
        
        # Fit the model
        model.fit(X_train_rfe, y_train)
        
        # Predict on the validation set
        y_val_pred = model.predict(X_val_rfe)
        
        # Calculate F1 Macro score
        score = f1_score(y_val, y_val_pred, average='macro')
        scores_list.append(score)
        
        if score > high_score:
            high_score = score
            best_features = n
    
    print(f"Optimum number of features: {best_features}")
    print(f"F1 Macro Score with {best_features} features: {high_score:.6f}")
    
    return best_features, high_score, scores_list

#Function to plot decision tree feature importance
def compare_feature_importances(X_train, y_train, figsize=(13, 5)):
    """
    Compares feature importances using Gini and Entropy criteria in a Decision Tree Classifier 
    and visualizes the results in a bar plot.

    Parameters:
    - X_train: Training feature set (DataFrame or array with column names).
    - y_train: Target labels for training.
    - figsize: Tuple specifying the figure size for the plot (default=(13, 5)).

    Returns:
    - zippy: DataFrame containing feature importances for Gini and Entropy.
    """
    # Calculate feature importances using Gini and Entropy criteria
    gini_importance = DecisionTreeClassifier().fit(X_train, y_train).feature_importances_
    entropy_importance = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train).feature_importances_
    
    # Create a DataFrame to store and organize the feature importances
    zippy = pd.DataFrame(zip(gini_importance, entropy_importance), columns=['gini', 'entropy'])
    zippy['col'] = X_train.columns  # Add column names
    
    # Melt the DataFrame for easier plotting with Seaborn
    tidy = zippy.melt(id_vars='col').rename(columns=str.title)
    tidy.sort_values(['Value'], ascending=False, inplace=True)
    
    # Plot the feature importances
    plt.figure(figsize=figsize)
    sns.barplot(y='Col', x='Value', hue='Variable', data=tidy)
    plt.title("Feature Importances: Gini vs Entropy")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend(title="Criterion")
    plt.show()
    
    return zippy

#This function is it to compare feature importances using Gini and Entropy criteria in a Random Forest Classifier and visualizes the results in a bar plot.
def compare_rf_feature_importances(X_train, y_train, figsize=(13, 5), random_state=42):
    """
    Compares feature importances using Gini and Entropy criteria in a Random Forest Classifier 
    and visualizes the results in a bar plot.

    Parameters:
    - X_train: Training feature set (DataFrame or array with column names).
    - y_train: Target labels for training.
    - figsize: Tuple specifying the figure size for the plot (default=(13, 5)).
    - random_state: Random state for reproducibility (default=42).

    Returns:
    - importances: DataFrame containing feature importances for Gini and Entropy.
    """
    # Calculate feature importances using Gini and Entropy criteria
    gini_importance = RandomForestClassifier(random_state=random_state).fit(X_train, y_train).feature_importances_
    entropy_importance = RandomForestClassifier(criterion='entropy', random_state=random_state).fit(X_train, y_train).feature_importances_
    
    # Create a DataFrame to store and organize the feature importances
    importances = pd.DataFrame({
        'gini': gini_importance,
        'entropy': entropy_importance,
        'col': X_train.columns
    })
    
    # Melt the DataFrame for easier plotting with Seaborn
    tidy = importances.melt(id_vars='col').rename(columns=str.title)
    tidy.sort_values(['Value'], ascending=False, inplace=True)
    
    # Plot the feature importances
    plt.figure(figsize=figsize)
    sns.barplot(y='Col', x='Value', hue='Variable', data=tidy)
    plt.title("Random Forest Feature Importances: Gini vs Entropy")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend(title="Criterion")
    plt.show()
    
    return importances


#This function is used to select top k features based on scores using Chi-square test.
def select_high_score_features_chi2_no_model(X_train, y_train, threshold=25):
    """
    Performs Chi-square test to select top k features based on scores.

    Parameters:
    - X_train: Training feature set (DataFrame).
    - y_train: Training target labels.
    - threshold: Number of top features to select (default=25).

    Returns:
    - high_score_features: List of top feature names based on Chi-square scores.
    - scores: Corresponding Chi-square scores of the selected features.
    """
    # Perform Chi-square test
    feature_scores = SelectKBest(chi2, k=threshold).fit(X_train, y_train).scores_

    # Select top features
    high_score_features = []
    scores = []
    for score, f_name in sorted(zip(feature_scores, X_train.columns), reverse=True)[:threshold]:
        high_score_features.append(f_name)
        scores.append(score)

    print(f"Top {threshold} features based on Chi-square scores:", high_score_features)
    print("Corresponding Chi-square scores:", scores)

    return high_score_features, scores

#This function is used to select top k features based on scores using MIC test.
def select_high_score_features_MIC(X_train, y_train, threshold=25, random_state=42):
    """
    Selects the top features based on Mutual Information Criterion (MIC).

    Parameters:
    - X_train: Training feature set (DataFrame or array).
    - y_train: Training target labels (array-like).
    - threshold: Number of top features to select (default=25).
    - random_state: Random state for reproducibility (default=42).

    Returns:
    - high_score_features: List of top feature names based on MIC scores.
    - scores: Corresponding MIC scores of the selected features.
    """
    # Calculate MIC scores
    feature_scores = mutual_info_classif(X_train, y_train, random_state=random_state)

    # Select top features based on MIC scores
    high_score_features = []
    scores = []
    for score, f_name in sorted(zip(feature_scores, X_train.columns), reverse=True)[:threshold]:
        high_score_features.append(f_name)
        scores.append(score)

    print(f"Top {threshold} features based on MIC scores:", high_score_features)
    print("Corresponding MIC scores:", scores)

    return high_score_features, scores







    

# Define a function to print metrics and plot a colorful confusion matrix
def metrics(y_train, pred_train, y_val, pred_val):
    # Print classification report for training data
    print('___________________________________________________________________________________________________________')
    print('                                                     TRAIN                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_train, pred_train))
    train_cm = confusion_matrix(y_train, pred_train)
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
    val_cm = confusion_matrix(y_val, pred_val)
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
