# ML-Project-Football-Prediction
Milestone Project for Intro to AI course


I. Introduction
   - This project is aimed to predict the result of home team in Premier League. 
   - Result is predicted based on the performance of both team in last 5 matches(including statistics like : shot on target, xG(expected goal), Pos(Possession),etc.., Ranking of both team in last season).
   - The problem is binary classification(i.e the result is divided into Win/Non-Win).
   - Try out different algorithms(including Gradient Boosting and Random Forest) to evaluate the performance of each one.
   - Finetune the hyperparameter to optimize the perfomance and avoid overfit.

II. Implementation
   * Main steps:
- Step1: Scraping data from the site "https://fbref.com/en/comps/9/Premier-League-Stats" to collect information about matches in EPL. In this project, we get the history matches in last 10 seasons.
- Step2: Clean data to get rid of noisy(missing values, unconsistent name,etc...)
- Step3: Perform Feature Engineering to extract usefull columns that helps predict the result
- Step4: Build model, try difference type of sampling(Balanced and Stratified) , using K-fold cross-validation to finetune the hypeparameter, test the performance on test set to choose the final model
- Step6: Implement U.I with html and CSS
- Step7: Deploy the model to localhost using Fast-API


III. Restrictions
- Bad performance on the first 5 match of each season since we mainly predict match result based on the performance of both team in lasst 5 match
- The process is still donee manually which is difficult to scale in production envionment(tasks can be automated by integrating pipeline, implement CI/CD to deploy to clouds, using Docker)



IV. How to run this project
- Clone this url repo: "https://github.com/thanhbinh911/ML-Project-Football-Prediction" to your local machine
- Run "conda env create -f environment.yml" in your terminal/cmd to reproduce the environment (we using conda to manage libray and enviroment in this project)
- in the cwd, run the folling command: 'uvicorn src.main:app --reload' to initiate the local sever
- Go the this URL: http://127.0.0.1:8000 

