{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3e6a5a41-ed27-4b16-8277-924e3c8028d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Prediction Result:  No Failure\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "# Load trained Random Forest model\n",
    "with open(\"../models/random_forest_model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "# Define feature names exactly as used in training\n",
    "feature_columns = ['device', 'metric1', 'metric2', 'metric3', 'metric4', \n",
    "                   'metric5', 'metric6', 'metric7', 'metric8', 'metric9']\n",
    "\n",
    "# Create input data with all features, setting metric9 to 0 (or another default value)\n",
    "input_data = pd.DataFrame([[0, 215630672, 55, 0, 52, 6, 407438, 0, 0, 0]],  \n",
    "                          columns=feature_columns)  # Ensure all features are present\n",
    "\n",
    "# Make prediction\n",
    "prediction = model.predict(input_data)\n",
    "\n",
    "# Show result\n",
    "print(\"\\nPrediction Result: \", \"Failure\" if prediction[0] == 1 else \"No Failure\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6c1dca54-d799-42c9-a3ed-3bf82f806628",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ All features are now aligned. Proceeding with prediction.\n",
      "\n",
      "Prediction Result:  No Failure\n"
     ]
    }
   ],
   "source": [
    "# Load dataset to verify feature columns\n",
    "df = pd.read_csv(\"../datasets/processed_maintenance_data.csv\")\n",
    "expected_features = df.drop(columns=['failure']).columns.tolist()\n",
    "\n",
    "# Check for missing columns in input_data\n",
    "missing_features = set(expected_features) - set(input_data.columns)\n",
    "if missing_features:\n",
    "    print(f\"⚠️ Missing Features: {missing_features}\")\n",
    "\n",
    "# Ensure all required features exist\n",
    "for feature in missing_features:\n",
    "    input_data[feature] = 0  # Fill missing features with default values\n",
    "\n",
    "print(\"✅ All features are now aligned. Proceeding with prediction.\")\n",
    "\n",
    "# Make prediction\n",
    "prediction = model.predict(input_data)\n",
    "print(\"\\nPrediction Result: \", \"Failure\" if prediction[0] == 1 else \"No Failure\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "585477fa-0963-4e8d-83ed-a78fea07d172",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
