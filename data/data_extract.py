from datasets import load_dataset
import json
def pubmed_extract(nb_samples = 1000000):
    pubmed = load_dataset('pubmed', streaming=True)
    processed_data = []

    for idx, entry in enumerate(pubmed['train']):
        if idx < nb_samples:
            abstract_text = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
            
            if isinstance(abstract_text, list):
                # Join list of text if multiple sections exist
                abstract_text = " ".join(abstract_text)
            elif isinstance(abstract_text, str):
                abstract_text = abstract_text.strip()
            else:
                print(f"Unexpected data type for abstract_text in entry {idx}: {type(abstract_text)}")
                continue  # Skip this entry
            if abstract_text:  # Skip empty abstracts
                processed_data.append({"id": idx, "text": abstract_text})
            else:
                continue  # Skip this entry
        else:
            # Save the processed abstracts to a JSON file
            output_file = "pubmed_abstracts.json"
            with open(output_file, "w") as f:
                json.dump(processed_data, f, indent=2)

            print(f"Processed data saved to {output_file}.")
            return processed_data
    
def load_pubmed_extracted(path):
    f = open(path,) 
    return json.load(f)
