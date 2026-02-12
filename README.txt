1. Question I had coming into this
- Based off of basic input information could a model 
predict if an individual is at risk for CRC Based
off their diet

2. Why does it matter
- A person could screen themselves and see if they
are categorized as someone who may be at risk.
This could be helpful for someone who wants to
be made aware of if their diet is problematic

3. data
- The data was downloaded from kaggle
- The data has hundreds of observations 
- The key features of the data include their
history, and details about their diet.

4. method
- I chose a random forest regressor because it 
excels at classifying data. It can get very powerful
as it uses several trees instead of just one.
- I evaluated it via confusion matrix and accuracy score
- The former to guage it quantitatively and the latter
for quick evaluation of how the model did
- I also cross validated this model to compare it
to other folds and prevent cherry picking and just ensure
I was getting the best representation of both may
model and my data
- I used a column transformer combined with
a pipeline to reduce data leakage and ease
the training process

5. Results
- The model scored .88 on its first test so
i was already able to implement it.
- There were several false negetives and this
may have to do with the fact that there are more
variables that have to do with a patients risk
level
    - This also may be the reason why a 3d PCA
    analysis appeared to only have one distinct
    cluster

6. Takeaways 
- I discovered that BMI was the biggest mover in
whether a person was at risk or not
- Vitamin intakes did not have much effect on A
persons risk level
