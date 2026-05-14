PenguinID: Individual Recognition via Wing Patterns
This repository provides a automated pipeline for identifying individual king penguins by analyzing the unique biometric "fingerprints" found in the patterns under their wings.

1. How the Pipeline Works
The current script uses a Re-Identification (Re-ID) approach rather than a traditional classifier. This means it doesn't need to be "retrained" every time you add a new penguin; it simply compares new wing patterns against your existing "gallery" of photos.

Data Sync: Automatically downloads your organized photos from Google Drive using gdown.

Feature Extraction: Utilizes MegaDescriptor-T, a vision transformer model specialized for wildlife, to turn wing patterns into unique numerical vectors.  

Pattern Matching: Uses Cosine Similarity to compare a new sighting against your database and identifies the most likely individual.

2. Setup & Installation
Ensure you have Python 3.9+ installed. Due to specific dependencies in ecology libraries, we use capped versions for numpy and scipy.

Install Dependencies:

Bash
pip install -r requirements.txt
Configure Drive:
Update the FOLDER_ID in the main script with your Google Drive folder ID.

3. Alternative Identification Options
Depending on your dataset size and technical requirements, you might consider these alternatives:

Option	Best For...	Pros/Cons
Wildbook (Wild Me)	Long-term research	
Pros: Industry standard, robust database. Cons: Complex setup (Docker).  

HotSpotter	Small datasets	
Pros: Excellent for 2D patterns (like wing spots) without needing ML training.  

LoFTR	Low-quality images	
Pros: High accuracy (up to 95%) by matching local features rather than global textures.  

YOLOv8 + Custom Re-ID	High-volume automation	
Pros: Can automatically find and crop the wing in a large landscape photo.  

4. Troubleshooting
If you encounter Dependency Conflicts, ensure your numpy version is below 2.0 and scipy is between 1.11.1 and 1.12. These specific versions are required for compatibility with common ecology tools like lifelines and pygam.