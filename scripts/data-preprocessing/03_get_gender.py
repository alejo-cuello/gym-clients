import unicodedata
import pandas as pd
import numpy as np

#  Imports are always put at the top of the file...
# [cite: 204] Imports should be grouped... Standard library, then third party.


def clean_text_helper(text: str) -> str:
    """
    Normalizes text by removing accents and standardizing encoding.
    """
    if isinstance(text, str):
        return (
            unicodedata
            .normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )
    return text


def process_gym_data(
    exercises_path: str,
    gender_path: str,
    output_path: str = "./data/03-preprocessed/exercises_v2.csv"
) -> pd.DataFrame:
    """
    Loads, cleans, and merges gym exercise data with gender data.
    """
    # --- 1. Load Data ---
    df_exercises = pd.read_csv(exercises_path)
    df_gender = pd.read_csv(gender_path)

    # --- 2. Clean Exercises Data ---
    # Clean prefixes using regex
    #  The preferred way of wrapping long lines is by using Python's
    # implied line continuation inside parentheses.
    df_exercises["date"] = (
        df_exercises["date"]
        .str.replace(r"^FECHA:?\s*", "", regex=True, case=False)
    )
    
    df_exercises["observations"] = (
        df_exercises["observations"]
        .str.replace(r"^OBSERVACIONES:?\s*", "", regex=True, case=False)
    )

    # Clean Client Names
    # We strip, lowercase, and remove the prefix first
    df_exercises["client_name_clean"] = (
        df_exercises["client_name"]
        .str.replace(r"^NOMBRE Y APELLIDO:\s*", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .apply(clean_text_helper)  # Apply the accent removal function here
    )

    # Filter out rows containing "fecha" in the name (from your original logic)
    df_exercises = df_exercises[
        ~df_exercises["client_name_clean"].str.contains("fecha", na=False)
    ]

    # --- 3. Clean Gender Data ---
    # Standardize the 'hombre' column to Gender M/F
    # Note: Avoid inplace=True in modern pandas.
    df_gender["hombre"] = df_gender["hombre"].fillna("n")
    df_gender["gender"] = np.where(df_gender["hombre"] == "s", "M", "F")
    
    # Ensure the join key is clean
    df_gender["client_name_clean"] = (
        df_gender["nombre"]
        .str.strip()
        .str.lower()
        .apply(clean_text_helper)
    )

    # Merge logic replaces the loop/sort logic.
    # We use a LEFT join to keep all exercise clients, adding gender where matches found.
    final_df = pd.merge(
        df_exercises,
        df_gender[["client_name_clean", "gender"]].drop_duplicates(subset="client_name_clean"),
        on="client_name_clean",
        how="left"
    )

    # --- 4. Output ---
    print(f"Processed {len(df_exercises)} rows.")
    # Check for missing genders (equivalent to your mismatch print loop)
    missing_gender_count = final_df["gender"].isna().sum()
    if missing_gender_count > 0:
        print(f"Warning: {missing_gender_count} rows did not match the gender database.")
        print(final_df[final_df["gender"].isna()])

    final_df.to_csv(output_path, index=False)
    
    return final_df


if __name__ == "__main__":
    # Define paths
    EXERCISES_FILE = "./data/03-preprocessed/exercises_v1.csv"
    GENDER_FILE = "./data/01-raw/clients_gender.csv"
    
    # Execute
    # [cite: 323] Always have the same amount of whitespace on both sides of a binary operator
    df_result = process_gym_data(EXERCISES_FILE, GENDER_FILE)
    
    # Display results
    print(df_result[["client_name_clean", "gender"]].sample(10))