import pandas as pd
import re

######## Lim inn andre endringer du har gjort med metadataen
df = pd.read_excel("helse_ordliste.xlsx")

# Rename columns 
df.columns = ['Health_Term', 'Video_File']

# Drop the first row 
df = df.iloc[1:].reset_index(drop=True)

# Function to remove trailing numbers from the "Health_Term" index
'''def clean_health_term(df: pd.DataFrame):
    df['Health_Term'] = df['Health_Term'].str.replace(r'\s\d+$', '', regex=True).str.strip()'''

# Function replace invalid characters for windows folder names
def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  
    filename = re.sub(r'[.-]+$', '', filename)
    return filename.strip()

# Apply cleaning functions
#clean_health_term(df)
df['Health_Term'] = df['Health_Term'].apply(sanitize_filename)

# Save the modified file 
df.to_excel('helse_ordliste_mod.xlsx', index=False)

