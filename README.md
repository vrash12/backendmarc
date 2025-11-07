# Flask Decision Tree Service

This backend service provides a Flask API for training, serving, and retraining a decision tree model for technology field recommendations. It is designed to work with the Laravel frontend and admin dashboard.

## Features
- Train and serve a decision tree model for tech field recommendations
- Expose REST API endpoints for predictions and retraining
- Integrate with CSV training data
- PDF export of decision tree
- Admin-triggered retraining via POST request

## Directory Structure
```
backend/
├── app.py                  # Main Flask app entry point
├── build_training_dataset.py # Script to build training dataset from source
├── dtree_service.py        # Decision tree model logic and API endpoints
├── model.pkl               # Saved model (pickle)
├── requirements.txt        # Python dependencies
├── run.py                  # Alternate entry point
├── seed_training_data.py   # Seed training data script
├── data/
│   └── training_data.csv   # Main training data
├── decision_tree.pdf       # Exported decision tree visualization
└── ...
```

## Setup
1. **Python Version**: Recommended Python 3.11+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Service**:
   ```bash
   python app.py
   # or
   flask run
   ```
   The service will start on `http://localhost:5001` by default.

## API Endpoints
### 1. `/predict` (POST)
- **Description**: Get tech field recommendations based on user answers.
- **Request Body**: JSON with answers
- **Response**: JSON with recommended field(s)

### 2. `/retrain` (POST)
- **Description**: Retrain the decision tree model using latest data.
- **Request Body**: None
- **Response**: JSON with retrain status, message, and duration

### 3. `/export_tree` (GET)
- **Description**: Download the decision tree as a PDF
- **Response**: PDF file

## Admin Integration
- The `/retrain` endpoint is triggered from the Laravel admin dashboard via AJAX POST.
- The result (success/error, duration) is shown in a modal.

## Development Scripts
- `build_training_dataset.py`: Build or update the training dataset
- `seed_training_data.py`: Seed initial data
- `dtree_service.py`: Core model logic and API

## Notes
- Training data is stored in `data/training_data.csv`
- Model is saved as `model.pkl` after training/retraining
- PDF export is generated as `decision_tree.pdf`

## License
MIT

## Authors
- IkedaLab-Daniel
- Mark Daniel Callejas

---
For questions or issues, please contact the project maintainers.
"# marcy" 
"# backendmarc" 
"# backendmarc" 
