# Turkish News Classification with BERTurk and Decision Agent

## 1. Project Overview

This project focuses on Turkish news classification using both classical machine learning and transformer-based deep learning.

The main goals were:

- To classify Turkish news articles into 10 categories
- To create a strong TF-IDF baseline
- To fine-tune BERTurk for Turkish news classification
- To analyze model behavior with detailed evaluation outputs
- To build a decision agent that routes news articles based on model confidence

The final system does not only predict a category. It also decides whether a news article should be automatically routed, checked by an editor, or manually classified.

---

## 2. Dataset

The project uses the Interpress Turkish News Category Dataset.

The dataset contains Turkish news articles from 10 categories:

dünya
ekonomi
eğitim
gündem
kültürsanat
magazin
sağlık
siyaset
spor
teknoloji

The raw dataset is not included in this repository because of size. It should be downloaded separately or prepared using the dataset scripts.

Expected prepared files:

data/
├── train_prepared.csv
├── val_prepared.csv
├── test_prepared.csv
└── label_map.json
3. Data Preparation

The raw dataset was cleaned and prepared before training.

The preparation steps included:

reading raw TSV/CSV data
cleaning text values
removing unnecessary spaces
computing word counts
filtering very short and very long texts
creating train, validation, and test splits
creating label map
saving prepared CSV files

After filtering and preparation, the final dataset sizes were:

Split	Number of Samples
Train	37,588
Validation	8,000
Test	9,555

Label mapping:

Category	Label
dünya	0
ekonomi	1
eğitim	2
gündem	3
kültürsanat	4
magazin	5
sağlık	6
siyaset	7
spor	8
teknoloji	9

The dataset is not perfectly balanced. Because of this, Macro-F1 was used in addition to accuracy.

4. Evaluation Metrics

The main metrics used in the project are:

Accuracy
Macro-F1
Weighted-F1
Precision
Recall
Confusion Matrix
Average Precision
Expected Calibration Error
Accuracy

Accuracy measures the overall ratio of correct predictions.

correct predictions / total predictions
Macro-F1

Macro-F1 calculates the F1-score for each class separately and then takes the simple average.

It gives equal importance to every class.

This is useful when classes are imbalanced.

Weighted-F1

Weighted-F1 also calculates F1 for each class, but averages them according to class support.

Classes with more samples have more effect on the final score.

5. TF-IDF + Logistic Regression Baseline

The first model was a classical machine learning baseline.

The pipeline was:

news text
→ TF-IDF vectorizer
→ Logistic Regression
→ category prediction

TF-IDF represents text by word and n-gram importance.

The main TF-IDF settings were:

Parameter	Value
max_features	100,000
ngram_range	(1, 2)
min_df	2
max_df	0.95
sublinear_tf	True

The model used both unigrams and bigrams.

Examples:

spor
ekonomi
süper lig
dış politika
sağlık bakanlığı

Logistic Regression was trained with:

class_weight="balanced"
solver="saga"
max_iter=1000

Class balancing was used because the dataset was not perfectly balanced.

6. TF-IDF Results

The TF-IDF + Logistic Regression model produced strong results.

Model	Validation Accuracy	Validation Macro-F1	Test Accuracy	Test Macro-F1	Test Weighted-F1
TF-IDF + Logistic Regression	0.8932	0.8928	0.8835	0.8856	0.8827

This result shows that Turkish news classification can be performed strongly even with classical machine learning methods.

The reason is that many news categories contain distinctive words. For example:

football, match, transfer → sports
market, economy, investment → economy
hospital, doctor, treatment → health
7. BERTurk Fine-Tuning

The second model was BERTurk.

BERTurk is a pretrained transformer model for Turkish. It was not trained from scratch in this project. Instead, it was fine-tuned for the 10-class news classification task.

The pipeline was:

news text
→ BERTurk tokenizer
→ input_ids + attention_mask
→ BERTurk sequence classification model
→ category prediction

Main settings:

Parameter	Value
Model	dbmdz/bert-base-turkish-cased
Number of classes	10
Max length	256
Batch size	8
Epochs	3
Learning rate	2e-5
Weight decay	0.01
Warmup ratio	0.1
Optimizer	AdamW
8. Tokenization

BERTurk does not process raw text directly. The text is first converted into token IDs.

Example:

bekleyiş → bekley ##iş
İntibak → İn ##ti ##bak

The tokenizer output includes:

Field	Meaning
input_ids	numerical token IDs
attention_mask	identifies real tokens and padding
labels	true category ID

Special tokens are also added:

Token	Meaning
[CLS]	beginning of sequence, used for classification
[SEP]	end of sequence

No aggressive stopword removal or stemming was used for BERTurk because Turkish suffixes and word forms can carry meaning. BERTurk’s tokenizer is designed to handle Turkish text using subword tokenization.

9. BERTurk Training

BERTurk was fine-tuned for 3 epochs.

Training results:

Epoch	Train Loss	Train Accuracy	Train Macro-F1	Val Loss	Val Accuracy	Val Macro-F1
1	0.6505	0.7934	0.8006	0.3493	0.8826	0.8820
2	0.3399	0.9012	0.9036	0.4405	0.8816	0.8813
3	0.2012	0.9462	0.9478	0.4775	0.8900	0.8895

The best model was selected according to validation Macro-F1.

10. BERTurk Test Results

Final BERTurk test results:

Metric	Value
Test Loss	0.5220
Test Accuracy	0.8864
Test Macro-F1	0.8881
Test Weighted-F1	0.8855

Comparison with TF-IDF baseline:

Model	Test Accuracy	Test Macro-F1	Test Weighted-F1
TF-IDF + Logistic Regression	0.8835	0.8856	0.8827
BERTurk	0.8864	0.8881	0.8855

BERTurk slightly outperformed TF-IDF.

The improvement was not very large because TF-IDF was already strong for this dataset. Turkish news categories often contain category-specific words, so word-based methods can perform well.

However, BERTurk still has an advantage because it can use contextual representations instead of only word frequency.

11. TF-IDF vs BERTurk

The main differences are:

Aspect	TF-IDF + Logistic Regression	BERTurk
Model type	Classical machine learning	Transformer-based deep learning
Text representation	Word and n-gram frequency	Contextual embeddings
Tokenization	Word / n-gram based	Subword tokenizer
Training	Trained from scratch on project dataset	Pretrained model fine-tuned
Context understanding	Limited	Stronger
Speed	Faster	Slower
Hardware need	CPU-friendly	GPU recommended
Interpretability	Easier	Harder

TF-IDF learns from the project dataset directly. BERTurk already has Turkish language knowledge and is adapted to the news classification task through fine-tuning.

12. Detailed Analysis

Additional analysis was performed after BERTurk training.

The detailed analysis included:

confusion pair analysis
category-level average precision
correct and wrong prediction examples
calibration analysis
reliability diagram
precision-recall analysis
common word analysis in wrong predictions

Detailed test summary:

Metric	Value
Total test samples	9,555
Correct prediction rate	0.8864
Wrong prediction count	1,085
Expected Calibration Error	0.0889
13. Confusion Pair Analysis

The most confused category pairs were:

True Category	Predicted Category	Count
kültürsanat	magazin	76
magazin	kültürsanat	55
kültürsanat	dünya	45
kültürsanat	ekonomi	43
ekonomi	teknoloji	39
kültürsanat	siyaset	37
kültürsanat	sağlık	37
siyaset	eğitim	36
siyaset	dünya	31
dünya	kültürsanat	29

The most difficult category was kültürsanat.

This is understandable because culture-art news can overlap with magazin, world news, economy, politics, and health topics.

14. Average Precision by Class

Average Precision results:

Category	Average Precision
spor	0.9943
gündem	0.9856
magazin	0.9613
teknoloji	0.9610
eğitim	0.9577
sağlık	0.9570
siyaset	0.9559
dünya	0.9422
ekonomi	0.9378
kültürsanat	0.8238

The strongest class was spor. Sports news usually contains highly distinctive words such as match, team, league, football, and transfer.

The weakest class was kültürsanat, which overlaps with many other categories.

15. Common Words in Wrong Predictions

The most common words in wrong predictions included:

türkiye
büyük
eğitim
başkan
olduğunu
ilk
yer
yıl
özel
arasında
önemli
devam
üniversitesi
genel
gelen
iyi
gün
farklı
zaman
türk
merkezi

Many of these words are general news terms. They are not strongly category-specific, which may explain why they appear often in wrong predictions.

16. Decision Agent

The final part of the project is the decision agent.

The agent uses the BERTurk prediction and decides what should happen to the article.

The possible decisions are:

Decision	Meaning
Automatic routing	Send article directly to the predicted editor desk
Editor-controlled routing	Suggest category but require editor confirmation
Manual classification required	Confidence is too low, human review required

The agent uses:

predicted category
confidence score
risk category information
Top-1 / Top-2 margin
17. Confidence and Top-1 / Top-2 Margin

Confidence is the model’s probability for its top prediction.

Top-1 / Top-2 margin is the difference between the highest and second highest prediction confidence.

Formula:

Top-1 / Top-2 margin = Top-1 confidence - Top-2 confidence

Example:

Top-1: spor, confidence = 0.92
Top-2: magazin, confidence = 0.05
Margin = 0.87

This means the model is confident and clear.

Another example:

Top-1: kültürsanat, confidence = 0.54
Top-2: magazin, confidence = 0.47
Margin = 0.07

This means the model is uncertain.

The agent requires both high confidence and enough Top-1 / Top-2 margin for automatic routing.

18. Agent Rules

The final agent rules were:

Risky categories: dünya, siyaset, gündem

For risky categories:
automatic routing threshold = 0.97

For other categories:
automatic routing threshold = 0.90

For automatic routing:
Top-1 / Top-2 margin must be at least 0.20

If confidence >= 0.65 but automatic conditions are not satisfied:
editor-controlled routing

If confidence < 0.65:
manual classification required

The risky categories were selected because world, politics, and agenda news can overlap.

19. Agent Results

The agent was evaluated on 1000 test samples.

Metric	Value
Sample count	1000
Correct predictions	893
Overall accuracy	0.8930
Average confidence	0.9715
Automatically routed articles	910
Human review required	90

Agent decision accuracy:

Agent Decision	Sample Count	Correct Count	Accuracy	Average Confidence
Automatic routing	910	847	0.9308	0.9960
Editor-controlled routing	58	29	0.5000	0.8182
Manual classification required	32	17	0.5313	0.5517

The agent achieved higher accuracy on automatically routed samples. This means the decision rules were useful for separating more confident predictions from more uncertain cases.

20. Limitations of the Agent

The agent is not perfect.

Some wrong predictions still had high confidence. This means confidence alone is not a perfect indicator of correctness.

For example, news involving politics and international relations may be difficult to separate between dünya and siyaset.

The Top-1 / Top-2 margin helps detect uncertainty, but it cannot catch every wrong high-confidence prediction.

21. Output Files

The outputs/ folder contains:

classification reports
confusion matrices
training curves
metrics.json
history.json
agent_summary.txt
agent_decisions.csv
top_confused_pairs.csv
average_precision_by_class.csv
reliability_diagram.png
precision_recall_curves.png
wrong_prediction_common_words.csv
tokenization_examples.txt

The trained model weights and raw dataset files are not included in this repository.

22. How to Run

Install requirements:

pip install -r requirements.txt

Prepare dataset:

python src/prepare_dataset.py

Train TF-IDF baseline:

python src/train_tfidf_baseline.py

Train BERTurk:

python src/train_berturk.py

Run decision agent:

python src/decision_agent.py --sample_size 1000

Run detailed BERTurk analysis:

python src/analyze_berturk_detailed.py

Show tokenization examples:

python src/show_tokenization_examples.py
23. Conclusion

This project showed that Turkish news classification can be performed successfully with both classical machine learning and transformer-based models.

TF-IDF + Logistic Regression achieved a strong baseline. BERTurk slightly improved the results by using contextual language representations.

The decision agent added a practical decision-making layer. Instead of only predicting a category, the system decides whether the article can be automatically routed or should be reviewed by a human.

Overall, the project demonstrates:

dataset preparation
classical NLP baseline
BERTurk fine-tuning
tokenization analysis
model evaluation
error analysis
confidence analysis
decision agent design

The final result is a complete Turkish news classification and routing system.
