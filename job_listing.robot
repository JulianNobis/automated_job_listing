*** Settings ***
Library             SeleniumLibrary
Library             Collections
Library             OperatingSystem
Library             CustomClickAndScrollLibrary.py    /opt/homebrew/bin/chromedriver

*** Variables ***
${URL}              https://www.karriere.at/
${JOB_TITLE}        Software Entwickler
${REGION_TITLE}     Wien

*** Keywords ***
Filter Jobs In Python
    [Documentation]    Filters jobs -> navigates job listing page and iterates over all (filtered) jobs -> model prediction
    Filter Jobs  ${URL}  ${JOB_TITLE}  ${REGION_TITLE}

*** Test Cases ***
Search And Collect Job Listings
    Filter Jobs In Python
