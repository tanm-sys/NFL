
import polars as pl
import glob

def analyze():
    print("--- Strategic Feature Analysis ---")
    
    # 1. Supplementary Data
    supp = pl.read_csv("supplementary_data.csv", null_values=["NA"], infer_schema_length=10000)
    
    targets = ["offense_formation", "receiver_alignment", "defenders_in_the_box"]
    for t in targets:
        if t in supp.columns:
            print(f"\n[{t}]")
            print(supp[t].value_counts().sort("count", descending=True).head(10))
            print(f"Unique Count: {supp[t].n_unique()}")
        else:
            print(f"[{t}] NOT FOUND")
            
    # 2. Tracking Data (Player Role)
    # Load first file
    files = glob.glob("train/input_*.csv")
    if files:
        df = pl.read_csv(files[0])
        print(f"\n[player_role]")
        print(df["player_role"].value_counts().sort("count", descending=True))
        print(f"Unique Count: {df['player_role'].n_unique()}")
        
        print(f"\n[player_weight]")
        print(df["player_weight"].describe())
        
    else:
        print("No tracking data found.")

if __name__ == "__main__":
    analyze()
