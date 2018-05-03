from sklearn.naive_bayes import MultinomialNB
# Instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
# Train the model 
nb.fit_transform(X_train, y_train)


# Perform classification on an array of test vectors X
predict_class = nb.predict(X)

# Return probability estimates for the test vector X
prob_class = nb.predict_proba(X)
