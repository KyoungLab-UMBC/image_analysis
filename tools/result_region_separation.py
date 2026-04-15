import pandas as pd
from pathlib import Path

# ==========================================
# Helper Function: Excel Column to Python Index
# ==========================================
def col2num(col_str):
    """Converts Excel column letter (e.g., 'A', 'B', 'AA') to 0-based index."""
    num = 0
    for c in col_str.upper():
        num = num * 26 + (ord(c) - ord('A')) + 1
    return num - 1

# ==========================================
# 1. Configuration & Variable Setup
# ==========================================
file_path = r"F:\Weekly progression\Summary\FBP\FBP level in Glucosome radius analysis 180 mM 37C 20260401.xlsx"  # Replace with your actual file path
read_sheet = "Coloc Sm Att_pc"       # Replace with the name of the sheet to read

# Use the col2num function to input Excel column letters directly
num_column = col2num("T")  # Used to be 1

# Define read columns based on the mathematical relationships you provided
read_column_a1 = col2num("S")  # Used to be 3
read_column_b1 = read_column_a1 + 1
read_column_a2 = read_column_a1 + 5
read_column_b2 = read_column_a1 + 8

# Define background columns
bg_column_a = read_column_a1 + 4
bg_column_b = read_column_a1 + 7

# ==========================================
# 2. Read cell_num array from Summary_num
# ==========================================
# pandas automatically treats row 1 as the header, so data reading starts from row 2
df_summary = pd.read_excel(file_path, sheet_name='Summary_num')
cell_num = df_summary.iloc[:, num_column].dropna().astype(int).tolist()

# ==========================================
# 3. Read Main Data and Assign Groups
# ==========================================
df_data = pd.read_excel(file_path, sheet_name=read_sheet)

# Create a Group ID for each row based on the counts in cell_num
group_ids = []
group_index = 1
for count in cell_num:
    group_ids.extend([group_index] * count)
    group_index += 1

# Ensure we only process rows that have a corresponding group
df_data = df_data.head(len(group_ids)).copy()
df_data['Cell_Group_ID'] = group_ids

# Extract the actual column names from their indices to make slicing easier
read_cols = [df_data.columns[read_column_a1], df_data.columns[read_column_b1], 
             df_data.columns[read_column_a2], df_data.columns[read_column_b2]]
bg_a_col = df_data.columns[bg_column_a]
bg_b_col = df_data.columns[bg_column_b]

# ==========================================
# 4. Calculate Group Averages and Classify
# ==========================================
# Calculate the average of bg_a and bg_b *within each specific cell group*
group_means = df_data.groupby('Cell_Group_ID')[[bg_a_col, bg_b_col]].transform('mean')

# Classify as High (>= mean) or Low (< mean)
# (Assuming "High" includes values exactly equal to the mean)
def classify_row(row, mean_a, mean_b):
    a_is_high = row[bg_a_col] >= mean_a
    b_is_high = row[bg_b_col] >= mean_b
    
    if a_is_high and b_is_high: return 'high_a_high_b'
    elif not a_is_high and b_is_high: return 'low_a_high_b'
    elif a_is_high and not b_is_high: return 'high_a_low_b'
    else: return 'low_a_low_b'

# Apply the classification
df_data['BG_Category'] = [
    classify_row(row, mean_a, mean_b) 
    for (idx, row), mean_a, mean_b in zip(df_data.iterrows(), group_means[bg_a_col], group_means[bg_b_col])
]

# ==========================================
# 5. Output to New Excel File
# ==========================================
p = Path(file_path)
output_filename = p.parent / f"{p.stem}_{read_sheet}_RegionSeperation.xlsx"
categories = ['high_a_high_b', 'low_a_high_b', 'high_a_low_b', 'low_a_low_b']

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    for cat in categories:
        # Filter data for the current background category
        cat_df = df_data[df_data['BG_Category'] == cat]
        
        # --- Sheet 1: All raw read columns for this category ---
        # Include Cell_Group_ID to maintain traceability
        out_raw_df = cat_df[['Cell_Group_ID'] + read_cols]
        out_raw_df.to_excel(writer, sheet_name=cat, index=False)
        
        # --- Sheet 2: Mean of the read columns per cell group ---
        # Calculate the mean of the read columns, grouped by their original cell group
        if not cat_df.empty:
            out_mean_df = cat_df.groupby('Cell_Group_ID')[read_cols].mean().reset_index()
        else:
            # Handle empty categories gracefully
            out_mean_df = pd.DataFrame(columns=['Cell_Group_ID'] + read_cols)
            
        out_mean_df.to_excel(writer, sheet_name=f"{cat}_mean", index=False)

print(f"Processing complete. Data saved to: {output_filename}")