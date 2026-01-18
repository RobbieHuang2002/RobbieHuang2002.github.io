---
title: "Experimenting with KNN and Webscraping Frustrations"
date: 2026-01-17 00:00:00
categories: [Supervised Learning]
tags: [ai]
---
# The Rabbit Hole

I've been watching the open machine learning course by Yury Kashnitsky, who devs deeper into the math behind the machine learning models that are used today. Most of the time spent was watching his explanation on Decision Trees and going over entropy and Information Gain (IQ) equations. 

Once he started covering KNN's I began to play around with datasets to see the usecases of the model such as creating a cancer detector (benign or malignant) given different numeric values achieving a 96% accuracy. 

This led me down a rabbit hole with text classication, specifically with news categories. Which led me to discover the curse of dimensionality and that sentences are not good as they created a 75k feature set of dimensionality which led to a 26% accuracy. 

I followed this up by going back to numbers and wanted to see if a KNN model could predict the winner of a UFC fight based off of significant strikes and areas of significant strikes. Which caused a huge headache as there were no publicly available datasets to use. So like any stubborn developer I decided to do it myself and scrape the internet...

# What I Built

I decided to scrape using beautifulSoup, grabbing all sorts of data from the each fight. In order to create the features I decided to use the difference of significant strikes. 

I then fed the data that I collected into a KNN model and trained it. I was able to achieve a 78% accruacy. 
![Descriptive alt text](/assets/img/Accuracy.png)


# Future work

Super cool to see myself go from collecting data to cleaning data to training a model. 

I think what's next is to break it down even further and look at significant strikes per round in a fight, as well as try using different models and comparing between them. 





