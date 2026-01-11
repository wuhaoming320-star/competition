# Course Review Sentiment Analysis System

## Overview
This project is a simple sentiment analysis system for course reviews using a Naive Bayes classifier.
It classifies reviews into Positive or Negative and visualizes keywords with a word cloud.

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- wordcloud

Install dependencies:
pip install pandas scikit-learn matplotlib wordcloud

## Dataset Format
Create a file named comments.csv:

text,label
The teacher explains very clearly,1
This course is very useful,1
Too much homework and very tiring,0
I cannot understand the lectures at all,0
The exam difficulty is reasonable,1
The course schedule is chaotic,0

label = 1: Positive
label = 0: Negative

## Running
python sentiment_system.py

## Output
- Classification report
- Prediction for new input
- Word cloud of positive reviews
