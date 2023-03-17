<h1 align="center">About me</h1>
<p align="justify">
I am a 4th year Data Science Bachelor/2nd year Engineering student (ING-2/Bac + 4) at CY Tech (ex-EISTI). 
</p>

<h1 align="jusitfy">Recent Projects and Internship work regarding Data Science, Machine Learning and Deep Learning </h1>



##  1. Air Quality in France Analytics and Prediction Dashboard
* Implemented an end-to-end data science and machine learning project in the IoT(internet of things) field through IBM's CRISP-DM( Cross-Industry Standard Process for Data Mining) methodology as part of a  months data science internship at Anagra, France.
* Scraped real-time regulated air pollutants data(using `Beautiful soup 4`) in France which is then transformed into an automated process where the data are stored in an SQL database.
* Data wrangling and cleaning were performed on the dataset to give an up-to-date measure of the air quality in various places and departments in France
* Applied machine learning and time series algorithms(using `sktime`) to make predictions about the air quality in different locations of France.
* Combined all steps to form a pipeline and the results were displayed on an interactive dashboard using Dash-plotly, providing live air quality measures, predictions, information/advice regarding various air qualities and an analytics page regarding the pollutants.
* Prepared a detailed functional and technical specifications document.
* Dashboard page 1 (Live air quality display): 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195636095-6c1243db-5e21-498e-917f-c20065bab129.png" alt="" width="600"/>
</p>

* Dashboard page 2 (Air quality and health information page): 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195638048-382cd0e3-0398-4e43-884a-106bd2f6144f.png" alt="" width="600"/>
</p>
* Dashboard page 3 (Prediction page):
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195636348-03663a4a-8ace-4eae-8350-257e4d6e6327.png" alt="" width="600"/>
</p>
* Dashboard page 4 (Further analytics : time series trends of each pollutant and overall air quality, pie chart representing air quality distribution over specified period page) :

<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195637441-0f386603-a736-4419-b4ac-ad990b0850ec.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195637588-921d0ce0-c052-4005-9bdf-489ecf97eec4.png" alt="" width="600"/>
  <img src="(https://user-images.githubusercontent.com/65787323/195637753-5e29b5d6-e686-46e4-81f1-643c79379ecc.png" alt="" width="600"/>

</p>


## [2. Restaurant reviews stance detection by performing text classification ](https://github.com/Sarveshmeenwa/Text-Classification---Restaurant-Reviews/blob/main/PA3c_15_Text_Classification.ipynb)

* Calculated an inter-annotator agreement, using `Cohen's Kappa, k` to assess consensus between annotators of dataset and to check reliability of data
* Pre-processing the features(text data) : convert all words to lowercase, remove all emoticons, substitute multiple spaces with single space, lemmatization, tokenization
* Select couple of models by creating a machine pipeline to assess performance 
* Hyper-parameter tuning using GridSearch-CV to optimize models and TF-IDF vectorizer 
* Evaluation of models using ROC_AUC curve, classification report(precision, recall and f1-score) where the best model is linear SVM with an AUC score of 0.994.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195652694-35b19a3a-fc48-4086-8afd-1cf13a0c4132.png" alt="" width="600"/>
</p>


* Used a confusion matrix to know more about the errors the system makes and selected couple of errors to identify why model did not predict accurately.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195653239-ea5582d9-4196-42a7-87c7-e62b66bab109.png" alt="" width="600"/>
</p>

* Performed feature importance to know which features the model considers important : The most important features leading to positive and negative reviews
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195652982-b6162bf2-6c7f-4537-81f1-472b300d0ddc.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195653111-3c7828e1-8854-4d29-810b-a5a5675572f8.png" alt="" width="600"/>
  
</p>


Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Text-Classification---Restaurant-Reviews/blob/main/PA3c_15-Text_Classification-Report%20(2).pdf)


## [3. Investigating Learner Disengagement In Massive Open Online Courses(MOOCs)](https://github.com/Sarveshmeenwa/Statistics/blob/main/stats_markdown.pdf)
* Performed feature engineering to identify types of learners based on their level of engagement resulting into 4 types of learners : completers, bystanders, disengaging and auditors). The learners are stratified based on the video consumption decile, quiz completion amongst other learning analytics.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195653454-981edef4-3fba-445e-a394-112f0e33a063.png" alt="" width="600"/>
  
</p>

* Performed a variety of statistical tests such as t-tests, one-way and two-way ANOVA, two-way ANOVA with interaction parameters,step-wise regressions, post-hoc test such as Tukey HSD to identify factors on video consumption.
* Identified factors influencing course completion by performing logistic regression and generating an Odds-Ratio table 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195657605-b81949c5-a9e2-4b7e-97cc-dd49c7e1651d.png" alt="" width="600"/>
</p>

* Comparison of video consumption behavior by the different types of learners(auditing,bystanders,completers and disengaged learners) with survival analysis and conducted log-rank tests to obtain Hazard Ratio(H.R)
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195659195-a40b910a-049f-4872-b399-2c7c8eb0e919.png" alt="" width="600"/>
</p>

* Differences in video consumption by gender,HDI and types of learners with hazard ratios.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195660315-6c5864ba-5422-4feb-bcd4-f773ad206ba4.png" alt="" width="600"/>
</p>


Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Statistics/blob/main/Intermediate_statistic_Sarvesh_Meenowa_final%20(2)%20(1)%20(1).pdf)

## [4. Car detection using Convolutional neural networks (CNNs)](https://github.com/Sarveshmeenwa/Image-Classification-Project/blob/main/PA5_15.ipynb)
* General goals : practicing the creation of neural networks with the Keras library; gaining some actual experience using CNNs and picture data; looking into CNN overfitting; studying data augmentation and transfer learning techniques to enhance your CNN
* Created a function to build the CNN with relevant output layer and loss function
* Evaluated the model using test score loss, and identify when model is overfitting by looking at the training and validation loss for each Epoch
* Applied data augmentation techniques to reduce risk of overfitting
* Applying a pre-trained convolutional neural network and VGG-16 as a feature extractor to perform transfer learning
* Visualizing the learned features in the VGG-16 model (the lowest level patterns that the model searches for in pictures are represented by the first convolutional layer. )

## [5. Social Network Analysis of Juries within France Higher Education](https://github.com/Sarveshmeenwa/Social-Network-Analysis-Project/blob/main/SNA_code_rmd_ipynb.pdf)
* Created multipartite graph of internal and external jury members based on the level of Jury 
to identify important jury members and partner institutions
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195664100-14f48105-73a4-4f68-8224-7120c5a903ed.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195664202-9b67141f-3aa0-4530-b300-33cffabd6c73.png" alt="" width="600"/>  
</p>

* Adjusted the node size to depend on centrality indicators to find more important jury members
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195664924-936db4cf-f324-41f7-bfe1-176b960564ed.png" alt="" width="600"/>
</p>
* Performed community detection using Louvain method to identify community(clusters) of jury members

* Used Association Rule Mining(Apriori Algorithm) on more than 100,000 PhD manuscripts to find links between disciplines.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195676391-1fdf6c9e-8d15-45c9-a92c-44bebeb44bb3.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195676535-b47403ae-d161-4ab0-a39b-58fab611a2fa.png" alt="" width="600"/>
</p>

* References between links : 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195676297-998b2220-9a7c-4c1c-8f3a-8d0dd381da67.png" alt="" width="600"/>
</p>

## [6. Text Mining on Doctoral Dissertation and NLP to perform POS-Tagging and generate language exercises](https://github.com/Sarveshmeenwa/NLP-and-text-mining/blob/main/NLP%20-%20Pos%20tagging.ipynb)
* Designed methodology to automatically detect the language of the thesis using stop words
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195678949-1694516e-2b5b-4c75-bf38-275e5627b323.png" alt="" width="600"/>
</p>
* Processed the data and used TF-IDF and cosine to assess similarity between the theses
* Used a bigram and a 3-gram model to find terms in Theses sorted by their TF-IDFs scores.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195680059-176c9c5e-f9ab-47b8-b9e9-c07d0913f2a0.png" alt="" width="600"/>
</p>

* Created a force-directed network based on cosine-similarity matrix of 10 theses 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195680616-3d98e6c3-6aae-4bba-9aa4-7fbd845dc7a2.png" alt="" width="600"/>
</p>

* Automatically created the necessary data for MCQ exercises (target language is English) with the following goals :
<p align="center">

         1.  identify 1 to 3 nouns in the sentences (among 4 words of the sentence)
        
         2.  identify 1 to 3 pronouns in the sentences (among 4 words of the sentence)
</p>

       
* Automatically create the necessary data for MCQ exer-cises (target language is German now) with the following goals:
<p align="center">

         1.  identify the gender of a noun
         
         2.  identify the case of a noun
</p>

       
* Used a  conjugator-based  strategy(mlconjug-3) to the sentence corpus to automatically generate exercise based on this tool
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195682173-52a247a5-f1c5-40f4-9cae-15ad26ff2f4c.png" alt="" width="600"/>
</p>

## [7. Data wrangling of French Theses Project](https://github.com/Sarveshmeenwa/Data-Wrangling-of-French-ThesesProject/blob/main/Final%20notebook%20(1).ipynb)
* Created a python web scraper to scrap the theses from theses.fr
* Investigate missing data and use an imputation technique to overcome this issue.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195731067-5b2ebe0a-2469-456e-a4ec-5110a11f8ef5.png" alt="" width="600"/>
</p>

* Performing exploratory data analysis(EDA) to check common issues such as :

        1. How did the proportion of defences at the first of january evolve over the years ?
        2. In the Author name, how common are homonyms? for e.g Cecile Martin
        3. Check for issues in the supervisor’s ID
* Identify outliers (such as finding supervisors who have mentored a surprisingly large number of PhD candidates) by thresholding the data over certain percentiles as well as checking if they are truly outliers or mistakes by further scraping the data to have some domain knowledge and cross-check scraped data with existing results.
* Investigated how the choice  of  the language of the manuscript evolved over the past decades :
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195731821-5950397e-7bcf-4766-89b7-dc861561bb47.png" alt="" width="600"/>
</p>

* Investigated what period of the year do PhD candidates tend to defend ?
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195731991-a0119394-3b4e-4750-bf0e-f6f91f931e43.png" alt="" width="600"/>
</p>

* Use a probable gender derivation library from python on PhD candidate’s first name to be able to plot the evolution of gender among PhD candidates over the past decades.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195732166-dc34a295-9595-4132-8bd6-f778691d0ebf.png" alt="" width="600"/>
</p>

Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Data-Wrangling-of-French-ThesesProject/blob/main/Data_wrangling_sarvesh_meenowa%20(2)%20(1).pdf)

## [8. Clustering of Phi (φ) and Psi (ψ) Angles in Proteins](https://github.com/Sarveshmeenwa/Clustering/blob/main/Final-G_15-Assignment_03-Clustering.ipynb)
* Used the K-means clustering method to cluster the phi and psi angle combinations in the data and techniques to assess optimal values of k such as : the elbow approach and Silhouette Coefficient. Then, used the optimal K value to plot the clusters
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195733082-024324db-5f00-4211-9bed-3bdfaa3aea86.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195733109-f6807623-d8e0-4ea1-aa4d-da5bcf54fe35.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195733132-4de040b1-3205-4d7f-94dc-ea8ae92319fd.png" alt="" width="600"/>
</p>



* Investigated methods to perform cluster validation such as using the Silhouette coefficient and by assessing the stability on subsets. This involves removing a random portion of points and check if the clustering does not change fundamentally.
![image](https://user-images.githubusercontent.com/65787323/195733304-a54069ba-09ec-434c-b6d6-890fb8fa76ee.png)

* Used DBSCAN to cluster the phi and psi angle combinations in the data by :


        1.Motivating the choice of the minimum number of samples in the neighbourhood for a point to be considered as a core point
        2.Motivating the choice of the maximum distance between two samples belonging to the same neighbourhood (“eps” or “epsilon”).
       
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195733502-7a1526b8-3e7b-440a-8c6c-5ecdcdad257b.png" alt="" width="600"/>
</p>
  
* Compared clusters obtained from K-means and DBSCAN
* Investigated whether the clusters found using DBSCAN are robust to small changes in the minimum number of samples in the neighbourhood for a point to be considered as a core point, and/or the choice of the maximum distance between two samples belonging to the same neighbourhood (“eps” or “epsilon”).

## [9. Dimensionality  reduction  and  clustering  techniques on artificially created data from dating apps](https://github.com/Sarveshmeenwa/Dimensionality-Reduction-Project/blob/main/dim_reduction_final_rmd.pdf)
* Looked for correlations among variables by using both parametric and non-parametric techniques (Pearson vs. Spearman for correlation between continuous variables).
* Perform a PCA on relevant continuous variables of the dataset and created a circle of correlations 
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195734468-52360edb-ff81-47c1-9007-e5a4d0a6fb7d.png" alt="" width="600"/>
</p>

* Used scree-plots with both eigenvalues and percentage of variance explained as diagnostic tool to select number of principal components to keep
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195734916-ba6be7a4-94de-4958-afff-5eb321b99109.png" alt="" width="600"/>
</p>

* Performed feature engineering by creating a PCA biplot and loadings table to find the latent variables.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195735105-e3ae9e40-d32c-461b-a3dd-d1e3258e9824.png" alt="" width="600"/>
</p>


* Performed MCA on categorical variables(as well as using scree-plots to diagnose the number of principal components to keep) and created biplots of the variables and variable categories to identify variables which are most correlated with each dimension.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195735503-ba3e4b85-e996-4993-876c-75e6f79b7bd9.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195735522-d2310155-5b7a-47e6-b04e-87ef200f1044.png" alt="" width="600"/>

</p>



* Performed a k-means clustering on principal components of the analysis and justified  the  choice  of  the  number of clusters, notably through a scree plot as well performed HCPC on continuous variables.

Full report link : [:open_file_folder:](https://github.com/Sarveshmeenwa/Dimensionality-Reduction-Project/blob/main/Dim_reduction_Sarvesh_Meenowa_final_report%20(2).pdf)

## [10. In Depth Understanding of Machine Learning Workflows and Decision-trees](https://github.com/Sarveshmeenwa/ML-Workflows-and-Decision-Trees/blob/main/PA1%2015%20_%20Introduction%20to%20ML%20workflow%20and%20decision%20trees.ipynb)

* Understanding ML workflows :


                Step 1: Read data
                Step 2: Cross-validate after training the baseline classifier.
                Step 3: Trying out several classifications and rerunning the cross-validation.
                Step 4: To enhance the performance of the multiple classifiers, tune their hyperparameters. For instance, the maximum depth of the decision tree classifier may be controlled by a parameter, and the number of layers and neurons in each layer of the neural network classifier can be adjusted.


* Investigated best split parameters of decision tree using 3 types of scores : information gain, majority sum and gini. Once the best scorer was found, the model was tuned with the best max_depth and also visualised to understand what is going on behind the scenes.
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/195736559-462aaa9a-6c1e-44ec-aae2-775c03e09144.png" alt="" width="600"/>
  <img src="https://user-images.githubusercontent.com/65787323/195736807-5f196b06-ac2a-4c74-b863-5e9c6a6762b5.png" alt="" width="600"/>

</p>

* Modified the criterion of decision tree regressor using variance reduction method
* Investigated overfitting,underfitting and generalization gap on the modified decision tree regressor by comparison the MSE (Mean Squared Error) against Max_depth of the decision trees. Then compared the output by repeating the same procedure using sklearn's decision tree regressor, therefore providing the following comparison : 

<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/201369297-e87f6970-059b-4ec4-91e5-465cba7efa78.png" alt="" width="500"/>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/201369663-10ee977a-fa4e-47fb-b7cd-6674ab1bbbf1.png" alt="" width="
500"/>
</p>

## [11. Investigation of Random Forests for tabular data](https://github.com/Sarveshmeenwa/Random_Forests/blob/main/PA2_15-Random_Forest.ipynb)

* Working with UCI's adult dataset to encode categorical features (non-numerical features) using one-hot encoding using sklearn's `DictVectorizer`.
* Created a pipeline that first transformed the list of dictionaries into a numerical matrix, and then we used this matrix when training the classifier)

          from sklearn.pipeline import make_pipeline
  
          pipeline = make_pipeline(
            DictVectorizer(),
            DecisionTreeClassifier()
          )

* Investigated underfitting and overfitting of random forest classifiers by modifying the ensemble size and max depth as well as answering the following questions 

  * What's the difference between the curve for a decision tree and for a random forest with an ensemble size of 1, and why do we see this difference?
  * What happens with the curve for random forests as the ensemble size grows?
  * What happens with the best observed test set accuracy as the ensemble size grows?
  * What happens with the training time as the ensemble size grows?

<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/201375030-e389b264-62f6-46e0-a45e-85c4ff045b25.png" alt="" width="600" />
</p>

* Explored the feature importances in random forest classifiers from the UCI's Adult dataset
<p align="center">
  <img src="https://user-images.githubusercontent.com/65787323/201375456-54a403b2-f32f-4cb6-b449-d46f5d74aaf5.png" alt="" width="600" />
</p>

## [12. Implementing linear classifiers : SVC and perceptron ](https://github.com/Sarveshmeenwa/Implementation-of-linear-classifiers/blob/main/PA_04_15.ipynb)

* Using OOP, the [Pegasos algorithm](https://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf) was implemented for training support vector classifiers, and its hinge loss function was used to classify the sentiment of customer reviews of music albums as positive or negative. Additionally, the log loss function and the logistic regression model were applied to the same data using OOP.

* Addressed the bottlenecks in the algorithm by the calculations in linear algebra such as computing the dot product, scale the weight vector, and add the feature vector to the weight vector by :
  * Substituting Scipy's linear algebra functions for the standard Numpy mathematical operations in order to make linear algebra operations
  * Using sparse vectors are used.
  * BLAS functions are used in place of the vector scaling operations.


## [12. Spam detection](https://github.com/Sarveshmeenwa/Spam-detection/blob/main/spam_detection.ipynb)

