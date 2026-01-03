import chromadb
import json
import os


if not os.path.exists("./db"):
    print("------ No database found. Please run ingest.py first.")
    exit()

client = chromadb.PersistentClient(path="./db")

try:
    collection = client.get_collection(name="resume_store")
    count = collection.count()
    print(f"----- Connected to database. Found {count} resume chunks stored.\n")
    if count == 0:
        print("--------- Database is empty.")
        exit()
except Exception as e:
    print(f"---------- Could not find collection. Error: {e}")
    exit()

data = collection.get(include=["metadatas"])

print("="*120)
print(f"{'FILENAME':<30} | {'CANDIDATE NAME':<20} | {'EXP':<5} | {'LOCATION':<15} | {'EDUCATION (Preview)'}")
print("="*120)

seen_files = set()

if data['metadatas']:
    for meta in data['metadatas']:
        source_file = meta.get('source', 'Unknown')
        
        if source_file not in seen_files:
            name = meta.get('name', 'Unknown')
            exp = meta.get('years_exp', 0)
            location = meta.get('location', 'Unknown')
            
            education = meta.get('education', 'Unknown')
            if len(str(education)) > 30:
                edu_preview = str(education)[:30] + "..."
            else:
                edu_preview = str(education)
            print(f"{source_file[:30]:<30} | {name[:20]:<20} | {str(exp):<5} | {location[:15]:<15} | {edu_preview}")
            seen_files.add(source_file)

print("="*120)
print(f"\n🔍 Total Unique Resumes: {len(seen_files)}")

if data['metadatas']:
    print("\n🔍 JSON Dump of Last Processed Resume Metadata:")
    print(json.dumps(data['metadatas'][-1], indent=2))