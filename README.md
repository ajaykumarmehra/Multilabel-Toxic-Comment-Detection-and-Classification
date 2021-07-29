# Multilabel-Toxic-Comment-Detection-and-Classification 

Toxic comments refers to hatred online comments classified as disrespectful or abusive towards individual or community. With a boom of internet, lot of users are brought to online social discussion platforms. These platforms are created to exchange ideas, learning new things and have meaningful conversations. But due to toxic comments many users are not able to put their points in online discussions. This degrades quality of discussion.

We have developed various Machine Learning models to check comment is toxic and then classify the comments into different categories to examine the type of toxicity. Dataset used is https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and select the best algorithms based on our evaluation methodology. Moving forward we seek to attain high performance through our machine learning and deep learning models which will help in limiting the toxicity present on various discussion sites.

---
There are quite a lot of evaluation metric for machine learning models. The problem involves highly unbalanced dataset. So, accuracy is not a well-suited performance measure. We
finally settled on the ROC curve and AUC score which give a very accurate picture of the performance of a discriminative model, Hamming loss and Log loss.
<p align="center">
  <img src="https://user-images.githubusercontent.com/32571522/127516019-8c2c77a7-cc88-4cf2-b2dc-9a14b2d78438.png" />
</p>

## Explore
Try it out!

### Structure
  1. Machine Learning folder contains all machine learning models that we have tried.
  2. WebApp folder contains Flask WebApp. For running it follow steps below (Manually)
     

  ### Manually
  To run it directly on your local machine, I suggest using [venv](https://docs.python.org/3/library/venv.html). 

  You need to first change your working directory to WebApp folder then execute following command.

      pip install -r requirements.txt

  You then need to change your working directory to Toxic_Comment_WebApp folder then execute following command.

      python3 app.py

  Now you can browse the webpage:
  http://localhost:5000/


This is how its look

https://user-images.githubusercontent.com/32571522/127515104-59667f9f-0815-4ded-a672-55e41c7a37d2.mp4

1. In webapp, we have only take LSTM Model with Glove and Word2Vec embedding as it has best result according to evaluation metric. 
2. For 2 comments - "you little piece of shit" and "enter your comment here" you can see the classification done by model.

Please feel free to suggest improvements
