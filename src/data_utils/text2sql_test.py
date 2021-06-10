from data_utils import AtisDataSQL, simplify_sql

%load_ext autoreload
%autoreload 2

dataset = AtisDataSQL(10)
train_loader = dataset.train_dataloader()
batch = next(iter(train_loader))

sql = 'SELECT DISTINCT FLIGHTalias0.FLIGHT_ID FROM AIRPORT_SERVICE AS AIRPORT_SERVICEalias0 , AIRPORT_SERVICE AS AIRPORT_SERVICEalias1 , CITY AS CITYalias0 , CITY AS CITYalias1 , DATE_DAY AS DATE_DAYalias0 , DATE_DAY AS DATE_DAYalias1 , DAYS AS DAYSalias0 , DAYS AS DAYSalias1 , FLIGHT AS FLIGHTalias0 WHERE ( ( ( ( ( FLIGHTalias0.ARRIVAL_TIME < FLIGHTalias0.DEPARTURE_TIME ) AND DATE_DAYalias0.DAY_NUMBER = day_number0 AND DATE_DAYalias0.MONTH_NUMBER = month_number0 AND DATE_DAYalias0.YEAR = year0 AND DAYSalias0.DAY_NAME = DATE_DAYalias0.DAY_NAME AND FLIGHTalias0.FLIGHT_DAYS = DAYSalias0.DAYS_CODE ) OR FLIGHTalias0.FLIGHT_DAYS = DAYSalias1.DAYS_CODE AND DAYSalias1.DAY_NAME = DATE_DAYalias1.DAY_NAME AND DATE_DAYalias1.YEAR = year0 AND DATE_DAYalias1.MONTH_NUMBER = month_number0 AND DATE_DAYalias1.DAY_NUMBER = day_number1 ) AND CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = "city_name0" AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE ) AND CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = "city_name1" AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE ) AND FLIGHTalias0.STOPS = stops0 ;'
simplify_sql(sql)