## Machine Learning Lab1

Logistic Regression

By Zhang Yongting 2022.9.27

### 1 Logistic Regression

You can refer to the class notes(Part 3)

### 2 Data

[Loan Data Set | Kaggle](https://www.kaggle.com/datasets/burak3ergun/loan-data-set)

### 3 Tasks & Requirements

#### 3.1 Tasks

If you choose the frame we supply

In `Logistic.py`, write your own Logistic Regression class

In `Load.ipynb`

1. Deal with NULL rows, you can either choose to drop them or replace them with mean or other value 
2. Encode categorical features
3. Split the dataset into X_train, X_test, y_train, y_test, also you can then use normalization or any other methods you want
4. Train your model and plot the loss curve of training
5. Compare the accuracy(or other metrics you want) of test data with different parameters you train with, i.e. learning rate, regularization methods and parameters .etc

If you choose to write this project from scratch, you need todo

1. Data cleaning, Data encoding, or any other process methods
2. Write your own Logistic Regression method in `Logistic.py`and train with`Load` dataset
3. Compare the accuracy(or other metrics you want) of test data with different parameters you train with, i.e. learning rate, regularization methods and parameters .etc

#### 3.2 Requirements

- **Do not** use sklearn or other machine learning library,  you are only permitted with numpy, pandas, matplotlib, and [Standard Library](https://docs.python.org/3/library/index.html), you are required to **write this project from scratch.**
- You are allowed to discuss with other students, but you are *not allowed to plagiarize the code**, we will use automatic system to determine the similarity of your programs, once detected, both of you will get **zero** mark for this project.

### 4 Submission

- Report

  - The Loss curve of one training process

  - The comparation table of different parameters

  - The best accuracy of test data

- Submit a .zip file with following contents(You don't need to submit `loan.csv`)

  --Loan.ipynb

  --Logistic.py

  --ReadMe.md

  --Report.pdf

- Please name your file as `LAB1_PBXXXXXXXX_NAME.zip`, **for wrongly named file, we will not count the mark**

- Sent an email to  ml_2022_fall@163.com with your zip file before deadline

- **Deadline:  2022.10.16 23:59:59** (Considering the National Day)

- For late submission, please refer to [this](https://gitee.com/Sqrti/ml_2022_f#%E4%B8%80%E5%85%B3%E4%BA%8E%E8%AF%BE%E7%A8%8B)