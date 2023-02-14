# Student-Withdrawal-Prediction-Model

*"Do we need to worry about students who are given alternate offers - are they dropping out more than normal?"*

To answer this, a Tensorflow neural net was trained on the withdrawal status of students who had been given offers other than the ones they wanted over the past 4 years. Data was augmented (for example - indicating if this university their first or insurance choice, or when the student handed in their application - look how many finished at 1am!), sanitised to comply with GDPR, and the minor class oversampled with [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) due to imbalanced classes. 

Future augmentation could include how close the application or acceptance date was to the relevant deadline, or the disparity between the university rankings of the applied course and the given course (as it may not be the course itself that has more withdrawals, but how much of a compromise the course would be to the student). Both aspects can be web scraped.

<p align="center">
<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096417-ac8b51b2-06b8-43fd-ae95-ecd34f2ea7ee.png" /> 

<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096421-fb73f162-f27d-4322-ad81-8097953d5df6.png" /> 

<img align="centre" alt="SQL" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216096422-27f2c420-b5a7-44a1-ae1b-b076198381ad.png" /> 
<br></p>

The resulting model achieved ~92% accuracy after tuning the number of training epochs and how many variables were fed into the network, and was applied on data for new students arriving in the current year. The below graphs highlight where withdrawals are predicted at varying levels of granularity (NOTE: this has been heavily anonymised for GitHub) 

<p align="center">
<img align="centre" alt="Campus Group_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098365-0bead2aa-9d02-4bcd-a0ac-ee195693ee04.png" /> 

<img align="centre" alt="Current College_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098368-8c5a7983-bdbc-4017-9f09-04e51e3f3891.png" /> 

<img align="centre" alt="Current Sub-Discipline_predicted_withdrawals" width="300px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098362-21f0e8f5-16ec-459c-bd8e-81347ee6e3ca.png" /> 
<br></p>

These graphs, along with an email report (abbreviated example below) containing the identification codes of the students predicted to withdraw are then automatically emailed to the relevant authority to advise which areas may need extra support. Python scripts are easily scheduled so this report can be tailored to suit the needs of the organisation.

<p align="center">
<img align="centre" alt="email_example" width="500px" style="padding-right:10px;" src="https://user-images.githubusercontent.com/122735369/216098815-9cbebe53-be1b-4411-a313-14975efd536d.png" /> 
<br></p>




