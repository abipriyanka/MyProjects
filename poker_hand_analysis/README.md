# Poker Hand Analysis System

A machine learning-based system designed to evaluate poker hands and predict outcomes using Random Forest and Neural Networks.

---

## Overview

This project aims to predict poker hand outcomes by building a complete pipeline that includes:
- Data collection and preprocessing
- Hand evaluation
- Model building using Random Forests and Neural Networks
- Evaluation and prediction

The system draws inspiration from advanced game theory concepts like Counterfactual Regret Minimization (CFR) and Deep CFR.

---

## Components

### 1. Data Collection and Preprocessing (`PokerHandDataset`)
- Load data from the UCI Poker Hand dataset.
- Convert numerical card representations into human-readable formats (e.g., "Ah" for Ace of Hearts).
- Engineer features:
  - Card rank and suit
  - Suited hole cards
  - Pocket pair indicator
  - Hand strength metrics
  - Player position and action encoding

### 2. Hand Evaluation (`PokerHandEvaluator`)
- Analyze hand strength using standard poker hand rankings.
- Rank hands from High Card to Straight Flush.
- Evaluate all possible 5-card combinations from 7 available cards.
- Handle special cases like Ace-5 straights.

### 3. Random Forest Model (`RandomForestPokerModel`)
- Build a Random Forest classifier using scikit-learn.
- Predict win/loss outcomes and calculate feature importance.
- Train using 100 trees (configurable).
- Visualize feature importance to understand key winning factors.

### 4. Neural Network Model (`NeuralNetworkPokerModel`)
- Build a deep learning model using TensorFlow/Keras.
- Model architecture:
  - 3 hidden layers (128, 64, 32 neurons) with ReLU activation
  - 30% dropout layers
  - Sigmoid activation output
- Train for 50 epochs using Adam optimizer and binary cross-entropy loss.
- Visualize training and validation performance.

---

## System Integration

### Workflow:
1. **Data Preparation**
   - Load and preprocess poker hand data.
   - Engineer poker-specific features.
   - Split into training and testing datasets.

2. **Model Training**
   - Train both Random Forest and Neural Network models.
   - Monitor and visualize model performance.

3. **Model Evaluation**
   - Assess models using accuracy, precision, and recall.
   - Analyze and visualize feature importance.

4. **Prediction**
   - Predict outcomes and win probabilities for new poker hands.

---

## Benefits of Using Two Models

- **Complementary Strengths:**
  - Random Forests are effective for feature importance, while Neural Networks capture complex, non-linear relationships.

- **Model Comparison:**
  - Enables validation through traditional machine learning vs deep learning approaches.

- **Ensemble Potential:**
  - Predictions from both models could be combined for even stronger performance.

---

## Reference to Deep Counterfactual Regret Minimization (Deep CFR)

This project draws inspiration from:
- **Deep Counterfactual Regret Minimization** by Noam Brown, Adam Lerer, Sam Gross, and Tuomas Sandholm.

Key insights:
- Deep learning allows decision-making without manual abstraction.
- Regret minimization techniques guide better strategy convergence in imperfect-information games.
- Directly using raw features (without abstraction) enhances model learning.

While this project primarily uses Random Forests and Neural Networks for hand evaluation, the understanding of CFR principles shaped the poker hand evaluation strategy.

---

## Tools and Technologies

- Python
- scikit-learn
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib / Seaborn

---



## Example

Imagine you have:

- **Hole Cards:** 10♠, J♠ (Ten and Jack of Spades)
- **Community Cards:** 7♠, 8♠, Q♦, 2♣, 5♠

The system evaluates this hand and identifies it as a **Flush**. Based on this, the model predicts a win probability to guide strategic decision-making.

---

## Final Note

This project is an early attempt and may have areas for improvement, but it demonstrates the integration of machine learning with traditional strategic games like poker.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

#Poker #MachineLearning #DeepLearning #GameTheory #AI #Python #DataScience

