# Sarvesh Meenowa Portfolio
I am a data science student in 4th year(M1/ING2) in France.

## [Project 1: Air Quality in France Analytics and Prediction Dashboard](https://github.com/Sarveshmeenwa/France-air-quality-dashboard-and-prediction)
* Implemented an end-to-end data science and machine learning project in the IoT(internet of things) field through IBM's CRISP-DM( Cross-Industry Standard Process for Data Mining) methodology as part of a 3 months data science internship at Anagra, France.
* Scraped real-time regulated air pollutants data(using `Beautiful soup 4`) in France which is then transformed into an automated process where the data are stored in an SQL database.
* Performed data wrangling and transformation were performed on the dataset to give an up-to-date measure of the air quality in various places and departments in France
* Applied machine learning and time series algorithms(using `sktime`) to make predictions about the air quality in different locations of France.
* Combined all steps to form a pipeline and the results were displayed on an interactive dashboard using Dash-plotly, providing live air quality measures, predictions, information/advice regarding various air qualities and an analytics page regarding the pollutants.
* Prepared a detailed functional and technical specifications document.
* Dashboard page 1 (Live air quality display): 
![image](https://user-images.githubusercontent.com/65787323/195636095-6c1243db-5e21-498e-917f-c20065bab129.png)
* Dashboard page 2 (Air quality and health information page): 
![image](https://user-images.githubusercontent.com/65787323/195638048-382cd0e3-0398-4e43-884a-106bd2f6144f.png)
* Dashboard page 3 (Prediction page):
![image](https://user-images.githubusercontent.com/65787323/195636348-03663a4a-8ace-4eae-8350-257e4d6e6327.png)
* Dashboard page 4 (Further analytics : time series trends of each pollutant and overall air quality, pie chart representing air quality distribution over specified period page) :

![image](https://user-images.githubusercontent.com/65787323/195637441-0f386603-a736-4419-b4ac-ad990b0850ec.png)
![image](https://user-images.githubusercontent.com/65787323/195637588-921d0ce0-c052-4005-9bdf-489ecf97eec4.png)
![image](https://user-images.githubusercontent.com/65787323/195637753-5e29b5d6-e686-46e4-81f1-643c79379ecc.png)

## [Project 2: Restaurant reviews stance detection by performing text classification ](https://github.com/Sarveshmeenwa/Text-Classification---Restaurant-Reviews/blob/main/PA3c_15_Text_Classification.ipynb)

* Calculated an inter-annotator agreement, using `Cohen's Kappa, k` to assess consensus between annotators of dataset and to check reliability of data
* Pre-processing the features(text data) : convert all words to lowercase, remove all emoticons, substitute multiple spaces with single space, lemmatization, tokenization
* Select couple of models by creating a machine pipeline to assess performance 
* Hyper-parameter tuning using GridSearch-CV to optimize models and TF-IDF vectorizer 
* Evaluation of models using ROC_AUC curve, classification report(precision, recall and f1-score)
* Used a confusion metrix to know more about the errors the system makes and selected couple of errors to identify why model did not predict accurately.
* Performed feature importance to know which features the model considers important


Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Text-Classification---Restaurant-Reviews/blob/main/PA3c_15-Text_Classification-Report%20(2).pdf)
