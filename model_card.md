# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Logistic Regression classifier built using the scikit-learn library in Python. It is configured with a random state of 42 and a maximum of 1,000 iterations to ensure reproducible training results.

## Intended Use

This model is intended to predict whether an individual's income exceeds $50,000 per year based on demographic census data. It was developed specifically for educational purposes to demonstrate the deployment of a machine learning pipeline using FastAPI and continuous integration.

## Training Data

The model was trained on the publicly available Census Income dataset sourced from the UCI Machine Learning Repository. The original dataset was processed by isolating the categorical features for one-hot encoding, and 80 percent of the total data was allocated to the training set.

## Evaluation Data

The model was evaluated on a test set comprising the remaining 20 percent of the Census Income dataset. This data was preprocessed using the exact same one-hot encoder and label binarizer that were fitted to the training data to ensure consistent feature dimensions.

## Metrics

The model's performance was evaluated using standard classification metrics, specifically focusing on precision, recall, and the F1 score. Upon testing, the Logistic Regression model achieved a Precision of 0.7338, a Recall of 0.5633, and an F1 score of 0.6374.

## Ethical Considerations

The dataset relies heavily on sensitive demographic features, including race, gender, and native country, to make financial predictions. Utilizing these features risks perpetuating historical biases, meaning the model should not be deployed in real-world loan approvals or employment screening without rigorous fairness audits.

## Caveats and Recommendations

During training, the underlying mathematical solver struggled to converge due to the unscaled numerical data. To improve future performance and reliability, it is recommended to implement a standard scaler for numerical features in the preprocessing pipeline or to explore tree-based algorithms like Random Forest that are less sensitive to feature scale.
