"""
Load data dari Supabase dan simpan ke data/raw/.
"""

import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from supabase import create_client, Client
from dotenv import load_dotenv


def get_supabase_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def load_all_data(client: Client, table_name: str) -> pd.DataFrame:
    all_data = []
    batch_size = 1000
    offset = 0

    while True:
        response = (
            client.table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not response.data:
            break
        all_data.extend(response.data)
        offset += batch_size

    return pd.DataFrame(all_data)


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig) -> None:
    print(f"Connecting to Supabase, table: {config.supabase.table}")

    client = get_supabase_client()
    df = load_all_data(client, config.supabase.table)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    os.makedirs(os.path.dirname(config.data.raw), exist_ok=True)
    df.to_csv(config.data.raw, index=False)

    print(f"Raw data saved to: {config.data.raw}")


if __name__ == "__main__":
    main()