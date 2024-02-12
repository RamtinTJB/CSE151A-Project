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

## Group Members

- Ramtin Tajbakhsh - rtajbakhsh@ucsd.edu
- Ali Mohammadiasl - amohammadiasl@ucsd.edu
- Welokiheiakea'eloa Gosline-Niheu - wgoslineniheu@ucsd.edu
