# Automated Job Listings Collection Project 
by Julian Nobis in November 2023.

## Project Overview
This project automates the collection of job listings from [karriere.at](https://karriere.at) using Robot Process Automation coupled with a Machine Learning model for text classification. The automation is initiated by `job_listing.robot`, which utilizes a ChromeDriver to navigate the website and search for part-time remote job listings in Vienna for "Software Entwickler". The data scraping is handled by `CustomClickAndScrollLibrary.py`, a custom Python library that is linked to the robot script. The ML component, implemented in `MachineLearningModule.py`, classifies the job titles and assigns a 'label' (1 for relevant, 0 for irrelevant) in the pandas dataframe to indicate the relevance of each job listing.

## File structure
- **job_listing.robot**: Utilize RPA for opening ChromeDriver, navigating to karriere.at, and entering specific search criteria.
- **CustomClickAndScrollLibrary.py**: Custom Python library to scrape job listing data and store in a list.
- **MachineLearningModule.py**: ML module using text classification to categorize job listings based on their relevance of column 'job_title'. Methods `train` for training the data and `predict` for the model's evaluation
- **job_listings.csv**: Training data for the ML algorithm. This data was scraped on Nov 2nd, 2023, by the same code as was used to generate the test data. Contains columns job_title and job_description. 
- **relevant_phrases.csv**: contains a list of job titles for classifying if a job title is relevant or not. If the job title is contained in the csv file then the job title is relevant, otherwise not relevant
- **relevant.csv**: Result after model has been applied to the test data. Contains the relevant job listings.
- **irrelevant.csv**: Result after model has been applied to the test data. Contains the irrelevant job listings.

## Additional information
After the robot code is finished, two figures are shown.
- Figure 1: Pie chart showing the proportion of relevant and irrelevant number of jobs (absolute numbers are included to give the pie chart a bit more credibility)
- Figure 2: Boxplot showing frequency of job title "Software Entwickler" in job titles by category "relevant" and "irrelevant"

It is suggested to have a look at the two csv files `relevant.csv` and `irrelevant.csv` to gain information on the relevant and irrelevant job listings.

## Documentation
This file gives a broad overview of the purpose of this project, the files and how to execute the code. Further information regarding the code is found inside the respective files as **comments**.

## Prerequisites
- Python 3.x
- Robot Framework
- Selenium WebDriver
- ChromeDriver
- Scikit-learn
- Pandas
- NumPy

## Execution
Call the robot script by executing `robot job_listing.robot` inside a terminal.
