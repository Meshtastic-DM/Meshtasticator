T = readtable('results3.csv');
Legend = T.RoutingType;
BroadcastDelays = T.BroadcastDelayMean;
SensorDelays = T.SensorDelayMean;
DMDelays = T.DMDelayMean;
BroadcastReliability = T.BroadcastReliabilityMean;
SensorReliability = T.SensorReliabilityMean;
DMReliability = T.DMReliabilityMean;
EnergyConsumption = T.EnergyConsumptionMean;
maxDelay = max([BroadcastDelays; SensorDelays; DMDelays]);
minDelay = min([BroadcastDelays; SensorDelays; DMDelays]);
data = [BroadcastDelays,SensorDelays, DMDelays,BroadcastReliability,SensorReliability,DMReliability,EnergyConsumption];

spider_plot(data, ...
    'axeslimits',[minDelay minDelay minDelay 0 0 0 0 ;maxDelay maxDelay maxDelay 1 1 1 700], ...
    'axesscaling',{'log','log','log','linear','linear','linear','linear'}, ...
    'axeslabels',{'Broadcast Delay(ms)','Sensor Delay(ms)','DM Delay(ms)','Broadcast Reliability','Sensor Reliability','DM Reliability','Enerygy Consumption (J)'} ,...
    'filloption', {'on', 'on'})
legend(Legend);

