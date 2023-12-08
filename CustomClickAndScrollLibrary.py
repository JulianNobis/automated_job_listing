# CustomClickAndScrollLibrary.py

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementClickInterceptedException
from MachineLearningModule import MachineLearningModel
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt  
from plotly.offline import plot
import pandas as pd
import time
import os

class CustomClickAndScrollLibrary:
    def __init__(self, driver_path):
        service = Service(executable_path=driver_path)
        options = Options()
        options.page_load_strategy = 'eager' # eager page load strategy to not focus on images, iframes, etc.
        self.driver = webdriver.Chrome(service=service, options=options)

    # method called from robot script -> navigates job listing page and iterates over all (filtered) jobs -> model prediction
    def filter_jobs(self, url, job_title, region_title):
        self.driver.get(url)

        # job title, for instance "Software Entwickler"
        search_input = self.driver.find_element(By.ID, 'keywords')
        search_input.send_keys(job_title)

        # location, for instance "Wien"
        region_input = self.driver.find_element(By.ID, 'locations')
        region_input.send_keys(region_title)

        # wait and close cookies
        self.wait_and_close_cookies_if_present()

        # click search button
        search_button = self.driver.find_element(By.XPATH, "//button[@class='m-jobsSearchform__submit m-jobsSearchform__submit--index']")
        search_button.click()

        # select home office checkbox
        self.click_checkbox_homeoffice()

        # select part-time filter
        self.click_job_filter_and_option()

        # click final search button after all filters have been applied
        self.click_search_submit_button()

        # scrape all the pre-filtered jobs and store in a list
        job_list = self.click_job_items()

        # convert data to a pandas df with columns 'job_title' and 'job_description'
        job_list = self.clean_data(job_list)

        # call machine learning code
        self.execute_machine_learning_code(job_list)

    # method iterates over all the filtered jobs and returns a list containing all the jobs
    def click_job_items(self):
        job_list = []

        # click "load more jobs" button until all available (filtered) jobs are present
        self.click_load_more_jobs_button()
        
        jobs_skipped = 0
        for index in range(len(self.driver.find_elements(By.XPATH, "//*[@class='m-jobsList__item']"))):

            job_cells = self.driver.find_elements(By.XPATH, "//*[@class='m-jobsList__item']")
            cell = job_cells[index]
            
            # Get title of job by accessing the first link element
            try:
                title_element = cell.find_element(By.TAG_NAME, "a")
                # scroll into view
                self.driver.execute_script("arguments[0].scrollIntoView(true);", cell)
                
                WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(title_element))
                title_element.click()
                time.sleep(0.2) # necessary for the page content to load -> otherwise trouble with loading (couldn't make it work with ECs so I had to resort to static sleeps)
                
                # Wait for the element to be present before continuing
                element_present = EC.presence_of_element_located((By.XPATH, "//*[@class='m-jobContent__jobText']"))
                WebDriverWait(self.driver, 10).until(element_present)
                # Get the whole job text
                job_text = self.driver.find_element(By.XPATH, "//*[@class='m-jobContent__jobText']").text

                # add job title to job text because sometimes the first words of the job text are not the title so it might not get correctly identified
                job_text = title_element.text + ";" + job_text
                # append to list
                job_list.append(job_text)
            except StaleElementReferenceException:
                # if element is stale -> skip to the next iteration
                jobs_skipped += 1
                continue
            
        print("Jobs skipped:", jobs_skipped)

        return job_list

    # method clicks "load more jobs" button until all relevant jobs are shown
    def click_load_more_jobs_button(self):
        while len(self.driver.find_elements(By.XPATH, "//*[@class='m-loadMoreJobsButton__button']")) > 0:
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts:
                try:
                    element_present = EC.element_to_be_clickable((By.XPATH, "//*[@class='m-loadMoreJobsButton__button']"))
                    WebDriverWait(self.driver, 20).until(element_present)

                    # Scroll the load more button into view
                    load_more_btn = self.driver.find_element(By.XPATH, "//*[@class='m-loadMoreJobsButton__button']")
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_btn)

                    load_more_btn.click()
                    print("Button was successfully clicked.")
                    time.sleep(0.5)
                    break  # If click succeeded, exit the loop
                except StaleElementReferenceException:
                    print("Caught StaleElementReferenceException, retrying...")
                    attempts += 1
                except ElementClickInterceptedException:
                    print("Caught ElementClickInterceptedException, retrying...")
                    # Optional: Add handling for the intercepting element here, e.g., closing a pop-up
                    attempts += 1
    
    # clicks the homeoffice checkbox
    def click_checkbox_homeoffice(self):
        checkbox = self.driver.find_element(By.ID, 'homeoffice_true_label')
        checkbox.click()

    # clicks the employment type (part-time)
    def click_job_filter_and_option(self):
        # Click the filter button
        filter_button = self.driver.find_element(By.ID, 'jobsFilterButton-Anstellungsart')
        filter_button.click()
        
        # Wait for the employment type option to be clickable after the filter has been clicked
        employment_option = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'employmentTypes[]_3961_3961_label'))
        )
        employment_option.click()

    # clicks submit button after all the filters have been set
    def click_search_submit_button(self):
        submit_button = self.driver.find_element(By.CLASS_NAME, 'm-jobsSearchform__submit')
        submit_button.click()

    def execute_machine_learning_code(self, job_list):
        # this file contains the job listings that were collected on Nov 2nd, 2023 by myself using the same algorithm and writing the pandas df to a csv
        file_name = 'job_listings.csv'
        # Check if the file does not exist
        if not os.path.exists(file_name): # get training data and store it in a csv
            # Save DataFrame to CSV only if the file doesn't exist
            job_list.to_csv(file_name, index=False)
        else: # there is training data available --> train and predict
            self.ml_model = MachineLearningModel() # ML model containing train() and predict() methods
            self.ml_model.train()
            new_predictions = self.ml_model.predict(job_list)
            
            job_list['is_relevant'] = new_predictions
            
            relevant_df = job_list[job_list['is_relevant'] == 1]
            irrelevant_df = job_list[job_list['is_relevant'] == 0]
            self.write_to_csv("relevant.csv", relevant_df)
            self.write_to_csv("irrelevant.csv", irrelevant_df)

            self.visualization(relevant_df,irrelevant_df)

    # method used to visualize the results, showing a pie chart and a boxplot
    def visualization(self, relevant_df, irrelevant_df):
        # Count the number of relevant and irrelevant jobs
        relevant_count = len(relevant_df)
        irrelevant_count = len(irrelevant_df)
        print("Number of relevant jobs:", relevant_count)
        print("Number of irrelevant jobs:", irrelevant_count)

        # Data for plotting
        labels = 'Relevant', 'Irrelevant'
        sizes = [relevant_count, irrelevant_count]
        colors = ['lightblue', 'lightgreen']
        explode = (0.1, 0)  # explode the first slice (Relevant)

        # Plotting the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Proportion of Relevant ({relevant_count}) vs. Irrelevant ({irrelevant_count}) Jobs')
        plt.show()

        # Original key term that was searched for in the initial search
        key_term = "software entwickler"
        # Count occurrences in both DataFrames
        relevant_count = self.count_key_term(relevant_df, key_term)
        irrelevant_count = self.count_key_term(irrelevant_df, key_term)

        # Data for plotting
        data = {
            'Category': ['Relevant', 'Irrelevant'],
            'Frequency': [relevant_count, irrelevant_count]
        }
        plot_data = pd.DataFrame(data)

        # Plotting
        sns.barplot(x='Category', y='Frequency', data=plot_data)
        plt.title(f"Frequency of '{key_term}' in Job Titles")
        plt.show()

    def count_key_term(self, df, key_term):
        return df['job_title'].str.lower().str.contains(key_term.lower()).sum()

    # method converts data to a pandas df with columns 'job_title' and 'job_description'
    def clean_data(self, job_list):
        col_title = "job_title"
        col_description = "job_description"
        rows = []
        for job in job_list:
            parts = job.split(";")
            title = parts[0]
            description = parts[1]
            rows.append({col_title: title, col_description: description})

        return pd.DataFrame(rows)

    # method to handle (close) cookie popup
    def wait_and_close_cookies_if_present(self):
        try:
            # Wait for the cookie button to be clickable, but up to 10 seconds
            wait = WebDriverWait(self.driver, 10)
            cookie_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[@class='onetrust-close-btn-handler onetrust-close-btn-ui banner-close-button ot-close-icon']"))
            )
            cookie_button.click()
            print("Cookie consent has been closed.")
        except Exception:
            # If the cookie button doesn't appear within 10 seconds, ignore and move on
            print("No cookie consent found within the time limit.")

    # method for writing the dataframes containing the relevant and irrelevant jobs
    def write_to_csv(self, file_name, df):
        df.to_csv(file_name, index=False)
