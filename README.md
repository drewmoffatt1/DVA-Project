# PJM Load Forecast Visualizer
## Load Forecasting using Weather Patterns

### Prerequisites
1. Tableau Desktop
2. Python 3.X

## To Use Application
### 1. To Link Tableau Viz to Data Sources
*Note: this just needs to be done once on initial setup unless you rename/move a folder afterward*
1. Clone or download repo
2. Once cloned/unzipped, open up the Tableau viz (```DVA-Project/PJM Load Forecast Visualizer.twb```)
3. On the ‘Extract Not Found’ window make sure ‘Locate the extract’ (default) is selected and click ‘OK’
    
    <img width="539" alt="Screen Shot 2022-11-16 at 4 59 38 PM" src="https://user-images.githubusercontent.com/116284163/202312998-734f0ddc-db7e-429b-a395-0de47c394d76.png">
4. In the ‘Locate Extract’ file browser navigate to ```DVA-Project/output``` and open the ```zone_mapping.csv+ (Multiple Connections).hyper``` file
    
    <img width="356" alt="Screen Shot 2022-11-16 at 5 00 21 PM" src="https://user-images.githubusercontent.com/116284163/202313086-417b8457-4baa-4c1f-9666-38df9a75fbd7.png">


### 2. To Update Forecast
1. Within the ```DVA-Project``` folder, execute ```forecastPJM.py``` file to create new PJM forecast output files
2. In the Tableau tool (```DVA-Project/PJM Load Forecast Visualizer.twb```), navigate to the top menu and select ```Data>Refresh All Extracts...``` and click 'Refresh' in the pop-up
    
    <img width="485" alt="Refresh-data" src="https://user-images.githubusercontent.com/116284163/202313386-f3e7c0e3-1204-4fda-acd2-1da6bbb809e6.png">
3. The map will automatically update and use the new forecast


