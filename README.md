# PJM Load Forecast Visualizer
## Load Forecasting using Weather Patterns

### Prerequisites
1. Tableau Desktop
2. Python 3.X

## To Use Application
### To View Tableau Viz
1. Clone or download repo
2. Once cloned/unzipped, open up the Tableau viz (```forecasting_tool/PJM Load Forecast Visualizer.twb```)
3. On the ‘Extract Not Found’ window make sure ‘Locate the extract’ (default) is selected and click ‘OK’
4. In the ‘Locate Extract’ file browser navigate to ```forecasting_tool/output``` and open the ```zone_mapping.csv+ (Multiple Connections).hyper``` file

### To Update Forecast
1. Navigate to the ```forecasting_tool``` folder and execute ```forecastPJM.py``` file to create new PJM forecast output files
2. In the Tableau tool (forecasting_tool/PJM Load Forecast Visualizer.twb), navigate to the top menu and select ```Data>Refresh All Extracts...``` and click 'Refresh' in the pop-up
3. The map will automatically update and use the new forecast

