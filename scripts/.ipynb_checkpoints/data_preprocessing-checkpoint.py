{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ffdc246d-43a2-4b47-8828-cad0ad1221df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Processed dataset saved in /datasets/processed_maintenance_data.csv\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from collections import Counter\n",
    "\n",
    "# Load dataset\n",
    "file_path = \"../datasets/predictive_maintenance_dataset.csv\"\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Convert categorical 'device' column to numeric\n",
    "encoder = LabelEncoder()\n",
    "if 'device' in df.columns:\n",
    "    df['device'] = encoder.fit_transform(df['device'])\n",
    "\n",
    "# Drop 'date' column if it exists\n",
    "if 'date' in df.columns:\n",
    "    df.drop(columns=['date'], inplace=True)\n",
    "\n",
    "# Apply feature scaling\n",
    "scaler = StandardScaler()\n",
    "numeric_cols = df.drop(columns=['failure']).columns\n",
    "df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n",
    "\n",
    "# Apply SMOTE to balance the dataset\n",
    "X = df.drop(columns=['failure'])\n",
    "y = df['failure']\n",
    "\n",
    "smote = SMOTE(sampling_strategy=0.2, random_state=42)\n",
    "X_resampled, y_resampled = smote.fit_resample(X, y)\n",
    "\n",
    "# Create balanced dataframe\n",
    "df_balanced = pd.DataFrame(X_resampled, columns=X.columns)\n",
    "df_balanced['failure'] = y_resampled\n",
    "\n",
    "# Save processed dataset\n",
    "df_balanced.to_csv(\"../datasets/processed_maintenance_data.csv\", index=False)\n",
    "\n",
    "print(\"✅ Processed dataset saved in /datasets/processed_maintenance_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2449f358-99a3-4242-8ac6-d4ede266fe05",
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
