# qanta-human-noise-injection
Attempting to improve accuracy of neural question-answering models by including incorrect human-submitted guesses in training data.

Collaborators: Jacob Bremerman, Sam Gollob, Megane Crenshaw, Asher Moldwin<br/>  Supervisors: Professor Jordan Boyd-Graber and Pranav Goel

<b>Introduction and Motivation</b><br/>
We originally started this project as a team research project for our Natural Language Processing course, taught by Professor Jordan Boyd-Graber at the University of Maryland, College Park.<br/> <br/>
Working within the QANTA question-answering framework, we decided to study whether including incorrect guesses that were submitted by humans in online practice sessions in our training data could improve our question-answering model, or at least speed up the training process.<br/><br/>
We noticed that during expo matches against humans, computers struggled more on certain types of questions, such as common link questions. They also often had guesses that were of a completely different category than the answer, which a human would never guess.  This led us to wonder whether including incorrect human guesses might at least prevent some of these computer-specific errors.	
We trained using data that contained human answers from Protobowl. We also modified our loss function in order to punish our model less when it got a wrong answer that was also an answer given by humans. <br/><br/>

<b>Preprocessing</b><br/>

In the "preprocessing" folder we have included a linux script to download the 5gb protobowl log file.We have also included the necessary scripts to extract the relevant data and store it in a JSON file accessible to our training script. These files condense the large protobowl.log file by consolidating repeat entries and associating human guesses with Wikipedia page titles. 
 We attempted to deal with the issue of misspelled and ambiguous human guesses by mapping each guess seen in the log file to the "real" answer for which it was most frequently marked correct:<br/>
![good mapping](https://github.com/amoldwin/qanta-human-noise-injection/blob/master/Images/mapping_good.png?raw=true)
<br/><br/>
As a next step, we would like to use context-based disambiguation to disambiguate guesses which were not mapped correctly using our current scheme:<br/>
![bad mapping](https://github.com/amoldwin/qanta-human-noise-injection/blob/master/Images/mapping_bad.png?raw=true)

 <br/>
 <b>Training and Model</b><br/>
We included the python file to train a DAN neural model. For faster running times, we recommend running this on a machine with access to a GPU, or otherwise using a cloud service with a GPU option. 
 <br/>
<b>Loss Function</b><br/>
To give the most weight to the correct answers but also take into account the human guesses, we used the following loss function:<br/>
 <p style="text-align: center;"><i>Loss = CrossEntropyLoss(model, answer) * (1 - HumanGuessFrequency)</i></p><br/>
 
 Next Steps:<br/>
 <b>Improve tools for error analysis: </b> We would like a quantitative measure of whether including the Protobowl.log data improves our accuracy on certain types of questions.  For example, we would like to see whether our model is making fewer egregious mistakes, such as guessing the entirely wrong category of answer (e.g. The Pacific ocean when the real answer was George Washington). For this we propose cross-referencing a type-of-entity database to check whether the answers are more likely to be of the right "type" when using our system.<br/><br/> Another option would be to use the wikipedia2vec vectorizer on our answer classes and use the cosine similarity as a measure of how far-off our guesses are from the true answers. <br/>The easiest measure of egregious errors could be: what percent of the time does our model give an answer that was never guessed for the current question? Or something similar.
 
<b>Disambiguating guesses mapping:</b> Incorporating context-based disambiguation to better map human guesses to answer classes. Alternatively, we could make a one-to-many mapping for this instead of one-to-one and calculate the model loss based on whether the guess class matched any of the possible mappings for that question's guesses.<br/>

<b>Experiment with Curriculum Learning and Contrastive Estimation</b> Possibly switch to negatively counting human-incorrect guesses after a certain accuracy is achieved. This threshold could be learned or tuned as a hyperparameter.<br/>

<b>Tune DAN learning rate, use proper vectorizer</b> Increasing the learning rate should give better results than we have achieved so far. Also use word2vec or glove instead of tfidfvectorizer. <br/>

<b>Try with bi-lstm and RNN</b> Obviously the purpose of this project is to explore a research question, not to achieve perfect question-answering accuracy. However, it might be reassuring to see if we can try this with a few other types of models to see if the effect is similar.

<b>Edit the QANTA Baseline System so it can use our model</b> Eventually we would like this to be a proper submission which can be evaluated using CodaLab at leaderboard.qanta.org<br/> 

 
