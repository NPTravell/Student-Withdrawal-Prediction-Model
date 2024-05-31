# Student-Withdrawal-Prediction-Model

*"Do we need to worry about students who are given less attractive offers - who is at risk of dropping out?"*

To answer this, ensemble machine learning was utilised. The data was augmented (for example - extracting the hour that the student handed in their application - look how many finished at 1am!) and sanitised for presentation

Future additional data points could include how close the application or acceptance date was to the relevant deadline (implying enthusiasm or organisation skills), or the disparity between leaderboard rankings of the course the student applied for and the course that they received (as it may not be the course itself that has more withdrawals, but how much of a compromise the course would personally be to the student). Both aspects can be web scraped.

<p align="center">
<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096417-ac8b51b2-06b8-43fd-ae95-ecd34f2ea7ee.png" /> 

<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096421-fb73f162-f27d-4322-ad81-8097953d5df6.png" /> 

<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096422-27f2c420-b5a7-44a1-ae1b-b076198381ad.png" /> 
<br></p>

An "ensemble" model is a logistic regression ‘meta model’ trained off 11 base models, use to account for the weaknesses of any individual model whilst utilising their strenfths. The result was an accuracy score of ~96% (and a Brier/loss score of 0.035). The below graphs highlight withdrawal results at varying levels of granularity.

<p align="center">
<img align="centre" alt="Campus Group_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098365-0bead2aa-9d02-4bcd-a0ac-ee195693ee04.png" /> 

<img align="centre" alt="Current College_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098368-8c5a7983-bdbc-4017-9f09-04e51e3f3891.png" /> 

<img align="centre" alt="Current Sub-Discipline_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098362-21f0e8f5-16ec-459c-bd8e-81347ee6e3ca.png" /> 
<br></p>

These graphs can then be sent with an automated email report (abbreviated example below) including the identity of the students predicted to withdraw to the relevant heads of departments, advising which areas may need extra support.

<p align="center">
<img align="centre" alt="email_example" width="500px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098815-9cbebe53-be1b-4411-a313-14975efd536d.png" /> 
<br></p>




