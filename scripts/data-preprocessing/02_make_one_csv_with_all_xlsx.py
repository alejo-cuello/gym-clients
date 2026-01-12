import pandas as pd
from pathlib import Path
import re

BASE_PATH = Path("csv")
OUTPUT_CSV = "exercises.csv"

rows = []

for excel_file in BASE_PATH.rglob("*.xlsx"):
    df = pd.read_excel(excel_file, header=None)

    # --- Metadata (adjust cell positions if needed) ---
    client = df.iloc[2, 1]          # "NOMBRE Y APELLIDO"
    raw_date = df.iloc[3, 1]        # FECHA
    observations = df.iloc[4, 1]

    # date = pd.to_datetime(raw_date, dayfirst=True)

    # --- Weeks column positions ---
    semana_1 = {
        "sets_1": 2,   # SER
        "reps_1": 3,   # REP
        "kg_1": 4,     # KG
        "rest_1": 5,    # PAUS
        "sets_2": 6,   # SER
        "reps_2": 7,   # REP
        "kg_2": 8,     # KG
        "sets_3": 9,   # SER
        "reps_3": 10,   # REP
        "kg_3": 11,     # KG
        "sets_4": 12,   # SER
        "reps_4": 13,   # REP
        "kg_4": 14     # KG
    }

    current_day = None

    for i in range(len(df)):
        cell_value = str(df.iloc[i, 1]).strip()

        # --- Detect day headers (DIA 1 ... DIA 5) ---
        day_match = re.match(r"DIA\s+(\d)", cell_value.upper())
        if day_match:
            current_day = f"Day {day_match.group(1)}"
            continue

        exercise = df.iloc[i, 1]

        if pd.isna(exercise) or current_day is None:
            continue

        cli = {
            "client_name": client,
            "date": raw_date,
            "day": current_day,
            "exercise": exercise,
            "observations": observations,
            "sets_1": None,
            "reps_1": None,
            "kg_1": None,
            "rest_1": None,
            "sets_2": None,
            "reps_2": None,
            "kg_2": None,
            "sets_3": None,
            "reps_3": None,
            "kg_3": None,
            "sets_4": None,
            "reps_4": None,
            "kg_4": None
        }

        for j in range(4):
            cli["sets_" + str(j+1)] = df.iloc[i, (semana_1["sets_" + str(j+1)])]
            cli["reps_" + str(j+1)] = df.iloc[i, (semana_1["reps_" + str(j+1)])]
            cli["kg_" + str(j+1)] = df.iloc[i, (semana_1["kg_" + str(j+1)])]
            if j == 0:
                cli["rest_" + str(j+1)] = df.iloc[i, (semana_1["rest_" + str(j+1)])]

        # Skip rows without real data
        if pd.isna(cli["sets_1"]) and pd.isna(cli["reps_1"]):
            continue

        rows.append(cli)

# --- Export ---
result = pd.DataFrame(rows)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"CSV generated successfully: {OUTPUT_CSV}")