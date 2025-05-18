# ML Noise Impact Analysis

A web application for analyzing the impact of noise in annotations on machine learning model training quality.

## Features

- Upload or select datasets
- Choose from various machine learning models
- Configure different types and levels of noise
- Train models with the specified configurations
- Visualize and analyze training results

## Technologies

- **Streamlit**: For the user interface
- **FastAPI**: For the backend API
- **scikit-learn**: For machine learning models
- **MLflow**: For experiment tracking
- **Docker**: For containerization

## Project Structure

The project follows a modular structure with clear separation of concerns:

- `frontend/`: Streamlit application
- `backend/`: FastAPI application
- `ml/`: Machine learning components
- `services/`: Business logic services
- `data/`: Data storage

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+

### Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yourusername/noise-impact-analysis.git
   cd noise-impact-analysis
   \`\`\`

2. Create a `.env` file from the example:
   \`\`\`bash
   cp .env.example .env
   \`\`\`

3. Build and start the containers:
   \`\`\`bash
   docker-compose up -d
   \`\`\`

4. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - MLflow UI: http://localhost:5000

## Usage

1. Start by selecting or uploading a dataset
2. Choose a machine learning model
3. Configure the type and level of noise
4. Train the model
5. Analyze the results

## Development

### Running Locally

1. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. Run the frontend:
   \`\`\`bash
   cd frontend
   streamlit run app.py
   \`\`\`

3. Run the backend:
   \`\`\`bash
   cd backend
   uvicorn main:app --reload
   \`\`\`
