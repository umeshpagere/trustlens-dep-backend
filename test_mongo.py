from dotenv import load_dotenv
load_dotenv()

from app.services.analysis_storage_service import _get_collection

def check_mongo():
    print("Testing MongoDB connection...")
    coll = _get_collection()
    if coll is not None:
        try:
            # Check ping
            coll.database.command('ping')
            print("✅ MongoDB Ping Successful! TLS handshake works.")
        except Exception as e:
            print(f"❌ Ping failed: {e}")
    else:
        print("❌ Could not get collection.")

if __name__ == "__main__":
    check_mongo()
