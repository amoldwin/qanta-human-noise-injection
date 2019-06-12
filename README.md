# qanta-human-noise-injection
Attempting to improve accuracy of neural question-answering models by including incorrect human-submitted guesses in training data.

Collaborators: Jacob Bremerman, Sam Gollob, Megane Crenshaw, Asher Moldwin<br/>  Supervisors: Professor Jordan Boyd-Graber and Pranav Goel

<b>Introduction</b><br/>
We originally started this project as a team research project for our Natural Language Processing course, taught by Professor Jordan Boyd-Graber at the University of Maryland, College Park.<br/> <br/>
Working within the QANTA question-answering framework, we decided to study whether including incorrect guesses that were submitted by humans in online practice sessions in our training data could improve our question-answering model, or at least speed up the training process.<br/><br/>
We noticed that during expo matches against humans, computers struggled more on certain types of questions, such as common link questions. They also often had guesses that were of a completely different category than the answer, which a human would never guess.  This led us to wonder whether including incorrect human guess might at least prevent some of these computer-specific errors.	
We trained using data that contained human answers from Protobowl. We also modified our loss function in order to punish our model less when it got a wrong answer that was also an answer given by humans. <br/><br/>
<b>Motivation: Noise Injection Techniques from Robotics</b><br/><br/>
As far as we could tell, this kind of intentional use of “poorly annotated” data (in our case the annotations would be non-expert Quizbowl guesses) has not been tried for NLP or question-answering tasks. However, we found an article titled “DART: Noise Injection for Robust Imitation Learning” which described a similar technique in the field of robotic imitation learning.  The authors of this paper describe the challenge of correcting machine-specific errors, due to the fact that there is often a lack of relevant training data. To guide the model toward better performance in these situations, specific forms of noise can be injected into the training data.
