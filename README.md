# Predicting the Housing Affordability Index in California Counties

## Dataset Links

- [California Housing Prices from 1990-2023](https://carorg.sharepoint.com/:x:/s/CAR-RE-PublicProducts/EcOe903EpvtJmUv1AaJFwp8BRTYd7-S3dKLWEH-edY6Oig?e=5h3lSl)
- [Median Income by Country from the Federal Reserve Economic Data](https://fred.stlouisfed.org/searchresults/?st=median%20income%20by%20county&t=ca&ob=sr&od=desc)

## Data Exploration

- [Link to the notebook](notebook.py)

General information about the "Housing Prices" dataset:
- Shape: (408, 111)
- Number of observations: 408
- Number of columns: 111

Out of these columns, only indices [0, 62] are valid and everything after that is empty. So we dropped all those columns from our dataframe.

Each row of the data represents the median prices of housing in California counties in a specific Year-Month, from Jan-1990 to Dec-2023.

#### Reconstruct the Date column
The "Mon-Yr" column in the original csv file contained seemingly random numbers, so we created a new column called "Dates" with actual Python `datetime` objects so we can plot our data as a time series. Here is the code for this:

```Python
from datetime import datetime

dates_arr = [""] * 408
for year_offset in range(34):
    for month in range(12):
        dates_arr[year_offset*12 + month] = datetime(year=1990+year_offset, month=month+1, day=1)

house_prices["Dates"] = dates_arr
```

#### Choosing 5 counties
Our dataset contains information from every single county in California. For the sake of this project, we decided to focus on 5 of them, but our approach can be generalized to any of the counties.

The counties that we chose are:
- San Diego
- Los Angeles
- San Francisco
- Orange
- Tulare

#### Plotting the prices and making observations
The code we used to plot the prices:
```Python
plt.rcParams["figure.figsize"] = (7,6)

for county in counties:
    plt.plot(house_prices["Dates"], house_prices[county], label=county)

plt.axvline(datetime(year=2007, month=7, day=1)) # August 2007
plt.axvline(datetime(year=2008, month=9, day=1)) # September 2008
plt.text(datetime(year=2001, month=1, day=1), 1.5e6, "Financial", size=12, c='b')
plt.text(datetime(year=2002, month=1, day=1), 1.4e6, "Crisis", size=12, c='b')

plt.axvline(datetime(year=2022, month=4, day=1), c='r')
plt.text(datetime(year=2016, month=1, day=1), 2e6, "COVID-19", size=12, c='r')

plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.show()
```

![Plot 1](imgs/Plot1.png)

**Some interesting observations we made**:
- During the financial crisis which started at around August 2007 and ended at around September 2008, there was a massive drop in housing prices. The blue vertical lines represent this timeline.
- The housing prices peaked in most couties during COVID which is represented by the red vertical line
- Tulare county is missing some data from ~1996 to ~2002

#### Separating each of the 5 counties into their own DataFrame
Since we might have to do different kinds of preprocessing steps on each county, it's convenient to have each county in a different dataframe.

After that, we can drop the rows that contain any NAs from the dataframes. Turned out that Tulare county was the only such county with 76 NAs.

#### Scatter plotting each county
Finally, we ran scatter on each county's housing prices to get a better of sense of them. The plots can be viewed in the notebook.

## Data Preprocessing

For data preprocessing we are planning do to the following tasks:
- Interpolate the missing data in the Tulare county (~1996-~2002)
- Either remove the prices from the housing bubble or interpolate that time frame to have more useful data. The reason for this is that a financial crisis is not something that happens all the time so it should be excluded from the training data.
- Normalize the data if needed.
- Preprocess the median income data for the 5 chosen counties

## Finishing up Data Preprocessing
- Interpolated the missing data points for Tulare county
- Changed the column names for each county for the median income data set and also changed the date to Python's datetime object
- Changed the data type of the median income from string to integer

Then we moved onto encoding the dates for the housing prices and the median income. Basically, what we did was that we changed the first date in the data set as day 0 and then we calculated every later date based on that. 

```py
for county in counties:
    county_dfs[county]["DATE_ENC"] = county_dfs[county]["Dates"] - county_dfs[county]["Dates"][0]
    county_dfs[county]["DATE_ENC"] = county_dfs[county]["DATE_ENC"].apply(lambda x: int(x.days))
```
Output:
``` 
        Dates  San Diego  DATE_ENC
0 1990-01-01   180484.0         0
1 1990-02-01   180714.0        31
2 1990-03-01   183701.0        59
3 1990-04-01   181567.0        90
4 1990-05-01   180794.0       120
```
As displayed above, the dates are calculated based on how many days away they are from day 0 of the dataset. Then we did the same thing on the median income dataset.   

![Plot 2]("imgs\BeforeInterpol.png")


![Plot 3]("imgs\AfterInterpol.png")

The first plot displayed above shows the median income data in its raw form (with missing data and `NAN`s) and the second plot shows the median income data after interpolation.

## Creating the First Model

For our first model we decided to do a linear regression. We ran a separate linear regression model on both the county median income as well as the housing prices. 

```py
linear_dict = {}

for county in counties:
    key = f'{county}_hp'
    linear_dict[key] = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(county_dfs[county]["DATE_ENC"].to_numpy().reshape(-1,1), \
                                                        county_dfs[county][county], test_size=0.2, random_state=15)
    linear_dict[key].fit(X_train, y_train)

    print(f'{county} Housing Price Linear Model:')
    y_test_pred = linear_dict[key].predict(X_test)
    y_train_pred = linear_dict[key].predict(X_train)
```

This is the code we used to create our model for the housing prices. Since we are doing five different counties, we decided to create a dictionary of models where we have a separate linear regression model for each county. Then we basically used the same code to create five separate linear regression models for the median income.

### Evaluating the model

We used mean squared error for our loss function. Our test mse was very close to the train mse. 

```
Orange Housing Price Linear Model:
	Train MSE:   13876777128.553144
	Test  MSE:   14064546197.695322
	Coefficient: 76.00820514368324
	Intercept:   87934.00198478042
```

Above is the output for the Orange county housing prices and as you can see the Train MSE and the Test MSE are relatively close, meaning that our model is not overfitting. Now, we can observe that the MSE is extremely high. But that is becuase we are working with really large numbers in the house prices data set. So the slightest error squared can lead to a huge number. 


![Plot 4]("imgs\OrangeLinReg")


The image above displays our linear regression model on the Orange County house prices dataset. We believe that it is a great fit for our dataset and it is not overfitting. 


![Plot 5]("imgs\OrangeMID")


This image shows the linear regression model ran on the Orange County median income data set. Again, the linear model looks quite good and does not look to be under or overfitting.


Lastly, after we created our models, we predict the value given an interest rate , the date, and the county.

We then plug those predicted values and the input into the House Affordibility Index equation and output the index. 

This is the code we used to compute the housing affordibility index:

```py
encoded_date_income = int((input_date - median_income_dict[input_county]["df"]["Dates"][0]).days)
encoded_date_housing = int((input_date - county_dfs[input_county]["Dates"][0]).days)

median_income = linear_dict[f'{input_county}_mi'].predict(np.array([encoded_date_income]).reshape(-1,1))
housing_price = linear_dict[f'{input_county}_hp'].predict(np.array([encoded_date_housing]).reshape(-1, 1)) 

pmt = housing_price * 0.8 * (interest_rate / 12) / (1-(1/((1+interest_rate/12)**360)))
qinc = pmt * 48
hai = median_income / qinc * 100
print(f'The Housing Affordability in {input_date.strftime("%B %Y")} is predicted to be {hai[0]:.4}%')
```


## Goals for next milestone

In conclusion for milestone 3, we can see that our linear regression model does a pretty good job of calculating the housing affordibility index. The regression lines look a good fit for our dataset and they don't see to be over or underfitting by much. The goal for the next milestone will be to train a Polynomial Regression model (can fit very well for data that isn't always fully linear) and a Lasso Regression model (it can help in identifying the most significant predictors). 



## Group Members

- Ramtin Tajbakhsh - rtajbakhsh@ucsd.edu
- Ali Mohammadiasl - amohammadiasl@ucsd.edu
- Welokiheiakea'eloa Gosline-Niheu - wgoslineniheu@ucsd.edu
