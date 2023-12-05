# Math789 Final Project - Fall 2023 ![Generic badge](https://img.shields.io/badge/Complete-black.svg)

*[Adler Viton](https://github.com/adlerviton), [Jeremy Tan](https://github.com/jeremymtan), [Katherine Tian](https://github.com/katherinetian540), [Andrew Kroening](https://github.com/andrewkroening)*

**Final Presentation is** [at this link](30_results/final_presentation.pdf)

**Full Technical Report** [can be found at this link](30_results/final_technical_report.pdf)

**Full Non-Technical Report** [can be found at this link](30_results/final_non_technical_report.pdf)

### Contents

- [Project Description](#project-description)
- [Technical Project Overview](#technical-project-overview)
- [Directories](#directories)

#### Data

Data for the project is available from:

- [Federal Reserve Daily Par Yield Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2023) for the years 1990-2023
- [S&P 500®](https://www.spglobal.com/spdji/en/indices/equity/sp-500) daily index values for the years 1990-2023
- [Bank of America](https://www.bankofamerica.com) for the information in the non-technical report

#### Project Description

This project repository combines a technical and non-technical team project for Math 789 (Fundamentals of Finance Business Models) from the Fall 2023 Semester at Duke MIDS. In the technical portion (this repository), we attempt to predict future interest rates from historical data. In the non-technical portion, we explore the structure of Bank of America and perform a simple risk analysis of its balance sheet.

#### Technical Project Overview

The technical project involved the use of neural networks, particularly a scikit-learn MLP Regressor, to simulate the interest rate yield curve daily for twenty years into the future. Generally, the models learned very well from historical data when training but struggled when we asked them to begin making predictions well into the future, past the end of the historical data. This slide provides a general idea of the experimental structure...

![alt text](40_docs/overview.png?raw=true)

and this one gives a more specific example of how we used a *lookback*, or historical, window to predict daily interest rates at the end of a *look-forward*, or prediction, interval...

![alt text](40_docs/experiment.png?raw=true)

In general, some of the results are more believable than others. We think that this might be our best two predictions, but we also have no way to validate the forecast beyond an eye test...

![alt text](40_docs/forecast.png?raw=true)

#### Directories

- [00_data](00_data) - This directory contains source data for the project
- [10_code](10_code) - This directory contains the code used for the project
- [20_intermediate_files](20_intermediate_files) - This directory contains the intermediate files used in the project, such as the models and forecast csvs
- [30_results](30_results) - This directory contains the project results
- [40_docs](40_docs) - This directory contains miscellaneous documents for the project
