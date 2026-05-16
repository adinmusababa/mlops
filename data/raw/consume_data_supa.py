from supabase import create_client
import os
from dotenv import load_dotenv

# load environment variables dari file .env
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

try:
    # test ambil data
    response = supabase.table(
        "customer_churn_raw"
    ).select("*").limit(1).execute()

    print("Koneksi berhasil!")

    # tampilkan data
    print(response.data)

except Exception as e:
    print("Koneksi gagal!")
    print(e)
    
# response = supabase.table(
#     "customer_churn_raw"
# ).select("*").execute()

# def get_data():
#     data = response.data
#     return data