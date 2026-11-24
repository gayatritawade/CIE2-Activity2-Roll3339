# SMS Spam Detection Using Logistic Regression

## Overview
This project classifies SMS messages into two categories:
- **Ham:** a normal message (Not Spam)
- **Spam:** an unwanted or spam message

This is a binary classification project. TF-IDF converts message text into numerical features, and Logistic Regression performs the classification.

## Problem Statement
The objective is to build a model that learns patterns in SMS text and predicts whether a new message is Spam or Not Spam.

## Objectives
- Load and explore the SMS Spam Collection dataset.
- Examine missing values and class distribution.
- Split the messages into training and testing sets.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate the model using Accuracy, Precision, Recall, F1-score, and a confusion matrix.
- Test the model with new SMS messages.

## Dataset
- **Dataset:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Input:** SMS text in the `message` column
- **Target:** `label` (`ham` or `spam`)
- **Total messages:** 5,574
- **Training messages:** 4,459
- **Testing messages:** 1,115

## Tools and Libraries
Python, Jupyter Notebook / Google Colab, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Machine Learning Approach
1. **TF-IDF Vectorizer:** converts text into numerical features based on word importance.
2. **Logistic Regression:** learns from labelled training messages and predicts the class.
3. **Pipeline:** combines TF-IDF and Logistic Regression. The vectorizer is fitted on training data only, helping prevent data leakage.
4. **Decision threshold:** 0.5 is used to classify a message based on its spam probability.

## Workflow
1. Import the required libraries.
2. Download and extract the dataset.
3. Load the messages into a Pandas DataFrame.
4. Explore missing values, dataset information, and class counts.
5. Visualize the ham and spam distribution.
6. Set `X` to the message column and `y` to the label column.
7. Split the data 80:20 using `random_state=42` and `stratify=y`.
8. Build a pipeline with `TfidfVectorizer` and `LogisticRegression`.
9. Train the model using `model.fit(X_train, y_train)`.
10. Predict test labels using `model.predict(X_test)`.
11. Calculate evaluation metrics and display the confusion matrix.
12. Test the model using new SMS messages.

## Model Performance
Reported test-set results:
- **Accuracy:** 97.40%
- **Precision (Spam):** 100.00%
- **Recall (Spam):** 80.54%
- **F1-score (Spam):** 89.22%
- **Decision threshold:** 0.5

Precision of 100% means all messages predicted as spam in this test set were actually spam. Recall of 80.54% means the model identified 80.54% of the actual spam messages; some spam messages were missed.

**ROC-AUC:** Calculate this in the notebook if it is required for your submission. Do not enter an estimated value.

## Example Predictions
| SMS message | Prediction | Spam probability |
|---|---|---:|
| Congratulations! You have won a free prize. Call now to claim. | Spam | 0.9369 (93.69%) |
| Hi, are we meeting for the project discussion today? | Not Spam | 0.0429 (4.29%) |

These are example predictions from the reported project run. Predictions are not guaranteed to be correct for every message.

## How to Run
1. Clone or download this repository.
2. Open `sms_spam_binary_classification.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook cells in order.
4. Allow internet access for the dataset download cell.
5. Review the evaluation metrics and sample predictions.

## Repository Contents
- `sms_spam_binary_classification.ipynb` — dataset loading, text processing, training, evaluation, and sample predictions.
- `README.md` — project documentation.

## Conclusion
This project demonstrates binary text classification using TF-IDF and Logistic Regression. On the reported test split, the model achieved 97.40% accuracy, 100% precision, 80.54% recall, and an 89.22% F1-score for the spam class.

## Limitations
- The model can make incorrect predictions.
- Performance may differ for messages in other languages, regions, or newer spam patterns.
- This educational model should not be used as the only spam-protection system.

## Author
Gayatri Tawade,AD3339.

