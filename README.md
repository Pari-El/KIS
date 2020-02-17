
# 1. Installations
You will need 
>Anaconda Navigator
>Jupyter Notebook 6.0.1
>Atleast Python 3. 

# 2. Project Motivation
I have been lucky enough to travel to a few countries in the past 5yrs, and one of the major aspects of travel which is accommodation. It is somewhat puzzling with the multifaceted aspects of travel which determine where i stay. 
I believe the influence of bookings is primarily based on the location and reviews and distance to city center. But,they may include the number of people that reviewed the place, ratings, price and location to city center.
In this research, I want to interrogate the Seattle Airbnb listings data to help me find some indicators and correlations between some of the above mentioned facets and ultimately predict based on location, property type and price, and ultimately see what factors result in the best customer traction for a potential Airbnb investment property. 

So as part of the research, i want to ask the below question and have a better understand of potential investment in a like setting:
> 1) Does the proximity to city center mean higher price?
> 2) I tend to find it easy to book a place with high amount of reviews, distance to city centre and price.lets investigate this reality
> 3) where is the best location, pricing and property type?

# 3. File Descriptions

## Data
the below file holds the data which was used to proceed with the above project motivation
> listings.csv

## Code
below is the Notebook which can be run on Jupyter
> SeattleAirbnbProject.ipynb
contains the code
> SeattleAirbnbProject.py

## Result
The Results to the questions posed can be explored in the Notebook 
> a plot graph of distance and price is can be generated to illustrate a visual corrolation, but further study shows that other factors may have played a part
> You will find that a R-squared of 1.85% is predicted for a 'number of reviews' response vector or ~1200 rows based on a 30% test split, for the distance, price and reviews quantitative variables. this implies that a very weak corrolation betten the study vectors, contrary to initial hypothesis.
> it can be noted based on the very last bar graph that the closer you are to the city, the property types are smaller and less independent. This generally maybe the case, but may vary based on the geography of the city

# 4. How to Interact with your project
The project can be interacted by running through the Notebook which addresses the questions posed initially.

# 5. Licensing, Authors, Acknowledgements, etc.
because this is my first time using Python, I would like to thank StackOverFlow contributors for their support, the list will be very long if i was to mention every contributor.
I would also like to acknowlege [kaggle]{https://www.kaggle.com/airbnb/seattle/data} for making available the seattle Airbnb Listings data

