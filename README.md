# qanta-human-noise-injection
Attempting to improve accuracy of neural question-answering models by including incorrect human-submitted guesses in training data.

Collaborators: Jacob Bremerman, Sam Gollob, Megane Crenshaw, Asher Moldwin<br/>  Supervisors: Professor Jordan Boyd-Graber and Pranav Goel

<b>Introduction and Motivation</b><br/>
We originally started this project as a team research project for our Natural Language Processing course, taught by Professor Jordan Boyd-Graber at the University of Maryland, College Park.<br/> <br/>
Working within the QANTA question-answering framework, we decided to study whether including incorrect guesses that were submitted by humans in online practice sessions in our training data could improve our question-answering model, or at least speed up the training process.<br/><br/>
We noticed that during expo matches against humans, computers struggled more on certain types of questions, such as common link questions. They also often had guesses that were of a completely different category than the answer, which a human would never guess.  This led us to wonder whether including incorrect human guess might at least prevent some of these computer-specific errors.	
We trained using data that contained human answers from Protobowl. We also modified our loss function in order to punish our model less when it got a wrong answer that was also an answer given by humans. <br/><br/>

<b>Preprocessing</b><br/>

In the "preprocessing" folder we have included a bash script to download the 5gb protobowl log file.We have also included the necessary scripts to extract the relevant data and store it in a JSON file accessible to our training script. These files condense the large protobowl.log file by consolidating repeat entries and associating human guesses with Wikipedia page titles. 
 We attempted to deal with the issue of misspelled and ambiguous human guesses by mapping each guess seen in the log file to the "real" answer for which it was most frequently marked correct:<br/><br/>
As a next step, we would like to use context-based disambiguation to disambiguate guesses which may not be mapped correctly using our current scheme:<br/>
![alt text](http://url/to/img.png)
 <br/>
 <br/>
