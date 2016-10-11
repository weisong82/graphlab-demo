import graphlab as gl
import datetime as dt
# household_data = gl.SFrame(
#       "https://static.turi.com/datasets/household_electric_sample/household_electric_sample.sf")
#
# household_data.save("household_data") ##

household_data = gl.SFrame("household_data")
print (household_data.head(10))

household_ts = gl.TimeSeries(household_data, index="DateTime")
print (household_ts.head(10))
