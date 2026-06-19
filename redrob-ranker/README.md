# OmniRank-X

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download and prepare the local model (requires internet access):
   ```bash
   python scripts/download_model.py
   ```

## Execution
Run the candidate discover and ranking pipeline:
```bash
python main.py
```
This writes the ranked candidates to `outputs/submission.csv`.

## Validation
Verify the output structure:
```bash
python scripts/validate.py
```
