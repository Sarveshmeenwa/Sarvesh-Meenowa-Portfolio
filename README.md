# [Sarvesh Meenowa Portfolio](https://sarveshmeenwa.github.io/Sarvesh-Meenowa-Portfolio/)
I am a data science student in 4th year(M1/ING2) in France.

## [Project 1: Air Quality in France Analytics and Prediction Dashboard](https://github.com/Sarveshmeenwa/France-air-quality-dashboard-and-prediction)
* Implemented an end-to-end data science and machine learning project in the IoT(internet of things) field through IBM's CRISP-DM( Cross-Industry Standard Process for Data Mining) methodology as part of a 3 months data science internship at Anagra, France.
* Scraped real-time regulated air pollutants data(using `Beautiful soup 4`) in France which is then transformed into an automated process where the data are stored in an SQL database.
* Data wrangling and cleaning were performed on the dataset to give an up-to-date measure of the air quality in various places and departments in France
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
![image](https://user-images.githubusercontent.com/65787323/195652694-35b19a3a-fc48-4086-8afd-1cf13a0c4132.png)*The best model is linear SVM with an AUC score of 0.994*


* Used a confusion matrix to know more about the errors the system makes and selected couple of errors to identify why model did not predict accurately.
![image](https://user-images.githubusercontent.com/65787323/195653239-ea5582d9-4196-42a7-87c7-e62b66bab109.png)

* Performed feature importance to know which features the model considers important
![image](https://user-images.githubusercontent.com/65787323/195652982-b6162bf2-6c7f-4537-81f1-472b300d0ddc.png)
*The most important features leading to positive reviews*
![image](https://user-images.githubusercontent.com/65787323/195653111-3c7828e1-8854-4d29-810b-a5a5675572f8.png)
*The most important features leading to negative reviews*


Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Text-Classification---Restaurant-Reviews/blob/main/PA3c_15-Text_Classification-Report%20(2).pdf)


## [Project 3: Investigating Learner Disengagement In Massive Open Online Courses(MOOCs)](https://github.com/Sarveshmeenwa/Statistics/blob/main/stats_markdown.pdf)
* Performed feature engineering to identify types of learners based on their level of engagement resulting into 4 types of learners : completers, bystanders, disengaging and auditors). The learners are stratified based on the video consumption decile, quiz completion amongst other learning analytics.
![image](https://user-images.githubusercontent.com/65787323/195653454-981edef4-3fba-445e-a394-112f0e33a063.png)

* Performed a variety of statistical tests such as t-tests, one-way and two-way ANOVA, two-way ANOVA with interaction parameters,step-wise regressions, post-hoc test such as Tukey HSD to identify factors on video consumption.
* Identified factors influencing course completion by performing logistic regression and generating an Odds-Ratio table 
![image](https://user-images.githubusercontent.com/65787323/195657605-b81949c5-a9e2-4b7e-97cc-dd49c7e1651d.png)
* Comparison of video consumption behavior between learners with survival analysis and conducted log-rank tests to obtain Hazard Ratio(H.R)
![image](https://user-images.githubusercontent.com/65787323/195659195-a40b910a-049f-4872-b399-2c7c8eb0e919.png)

*Video consumption behaviours by the different types of learners(auditing,bystanders,completers and disengaged learners)*

![image](https://user-images.githubusercontent.com/65787323/195660315-6c5864ba-5422-4feb-bcd4-f773ad206ba4.png)

*Differences in video consumption by gender,HDI and types of learners with hazard ratios.*


Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Statistics/blob/main/Intermediate_statistic_Sarvesh_Meenowa_final%20(2)%20(1)%20(1).pdf)

## [Project 4: Car detection using Convolutional neural networks (CNNs)](https://github.com/Sarveshmeenwa/Image-Classification-Project/blob/main/PA5_15.ipynb)
* General goals : practicing the creation of neural networks with the Keras library; gaining some actual experience using CNNs and picture data; looking into CNN overfitting; studying data augmentation and transfer learning techniques to enhance your CNN
* Created a function to build the CNN with relevant output layer and loss function
* Evaluated the model using test score loss, and identify when model is overfitting by looking at the training and validation loss for each Epoch
* Applied data augmentation techniques to reduce risk of overfitting
* Applying a pre-trained convolutional neural network and VGG-16 as a feature extractor to perform transfer learning
* Visualizing the learned features in the VGG-16 model (the lowest level patterns that the model searches for in pictures are represented by the first convolutional layer. )
