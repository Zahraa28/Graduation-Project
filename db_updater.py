import json
from pathlib import Path

METADATA_PATH = Path("/app/metadata_flat.json")

def update_local_metadata(brand, model, new_details):
    """Updates the flat JSON file with new info discovered by Serper/Groq."""
    if not METADATA_PATH.exists():
        return
        
    records = json.loads(METADATA_PATH.read_text())
    updated = False
    
    for rec in records:
        # Match the car we just found on the web
        if rec.get("brand") == brand and rec.get("model") == model:
            # Fill in the blanks if they were "Not Available"
            for key, value in new_details.items():
                if rec.get(key) in ["—", "Not_Available", ""]:
                    rec[key] = value
                    updated = True
                    
    if updated:
        METADATA_PATH.write_text(json.dumps(records, indent=2))
        print(f"Successfully learned new data for {brand} {model}")