About Dataset
WADI.csv

This dataset simulates high-dimensional industrial sensor data collected from multiple process units, representing real-time measurements from sensors, actuators, and control systems. Each record captures timestamped process variables such as pressure, flow rate, level, and valve status.

The dataset contains both normal and abnormal (attack) conditions. Attack instances simulate anomalies such as sudden pressure spikes, flow inconsistencies, or sensor malfunctions, reflecting potential faults or cyber-physical attacks in an industrial environment.

It is designed for developing and evaluating anomaly detection, fault diagnosis, and system monitoring models in industrial and IoT-based systems.

Key Features:

500 time-series records with realistic process variations

120+ sensor and actuator variables

Includes derived operational parameters

Binary target column: Attack (0 = Normal, 1 = Anomaly)

Suitable for anomaly detection, clustering, and pattern analysis

Column Description

Date – The calendar date when the sensor data was recorded.

Time – The exact time of each data reading.

1_AIT_001_PV – 1_AIT_005_PV – Analog input transmitters recording process variables such as temperature, flow, or pressure in Unit 1.

1_FIT_001_PV – Flow indicator transmitter showing the measured flow rate.

1_LS_001_AL, 1_LS_002_AL – Level switch alarms that indicate high or low tank levels.

1_LT_001_PV – Level transmitter providing liquid level measurements in a tank.

1_MV_001_STATUS – 1_MV_004_STATUS – Motorized valve status indicators (open/close state).

1_P_001_STATUS – 1_P_006_STATUS – Pump status signals indicating whether each pump is running or stopped.

2_DPIT_001_PV – Differential pressure indicator transmitter measuring pressure difference in Unit 2.

2_FIC_xxx_PV / CO / SP – Flow control loop variables:

PV: Process value (current flow rate)

CO: Controller output (control signal)

SP: Setpoint (desired flow rate)

2_FIT_001_PV – 2_FIT_601_PV – Flow transmitters from multiple process lines or sections.

2_LS_xxx_AH / AL – Level switch alarms for tanks or reservoirs (AH = Alarm High, AL = Alarm Low).

2_MV_xxx_STATUS – Motorized valve operational states for Unit 2.

2_P_xxx_STATUS / SPEED – Pump operational status and motor speed values.

2_PIC_003_PV / CO / SP – Pressure control loop readings showing current, output, and setpoint values.

2_PIT_001_PV – 2_PIT_003_PV – Pressure indicator transmitters measuring real-time line pressures.

2_SV_xxx_STATUS – Solenoid valve open/close status signals.

2A_AIT_xxx_PV / 2B_AIT_xxx_PV – Auxiliary analog input sensors from parallel or backup systems.

3_AIT_001_PV – 3_AIT_005_PV – Process variable readings for Unit 3 (temperature or pressure transmitters).

3_FIT_001_PV – Flow transmitter monitoring flow rate in Unit 3.

3_LS_001_AL – Level alarm in Unit 3 tank.

3_LT_001_PV – Level transmitter showing tank fluid level in Unit 3.

3_MV_xxx_STATUS – Motorized valve operation signals in Unit 3.

3_P_xxx_STATUS – Pump operation states (on/off) for Unit 3.

LEAK_DIFF_PRESSURE – Computed pressure difference used to detect potential leaks or blockages.

PLANT_START_STOP_LOG – Indicates plant operational state (1 = Startup/Active, 0 = Stop/Inactive).

TOTAL_CONS_REQUIRED_FLOW – Required or expected total process flow across the system.

Attack – Target column representing the system condition:

0 = Normal operation

1 = Attack or anomaly detected

SWAT.csv

About Dataset
This dataset contains multivariate time-series readings collected from multiple industrial process and monitoring units across different control levels. It is designed for research and experimentation in anomaly detection, system monitoring, and fault diagnosis in high-dimensional environments.
The dataset includes 1000 rows and 53 columns, where 52 columns represent numerical process variables, sensor readings, or control parameters, and 1 column (“Normal/Attack”) represents the class label indicating system state.
Key Features (Columns):

P206, DPIT301, FIT301, LIT301 – Core process and flow indicators representing pressure, differential pressure, and flow rate.

MV301–MV304 – Motor valve measurements from different control loops.

P301, P302 – Pressure sensors in secondary circuits.

AIT401, AIT402, FIT401, LIT401 – Analyzer, flow, and level indicators in subsystem 401.

P401–P404, UV401 – Monitoring and utility valve signals.

AIT501–AIT504, FIT501–FIT504 – Analyzer and flow indicators for the fifth subsystem.

P501, P502, PIT501–PIT503 – Pressure indicators and transmitters in high-sensitivity circuits.

FIT601, P601–P603 – Final-stage flow and pressure indicators.

Sensor_1 – Sensor_17 – Additional monitoring features representing auxiliary control readings and derived measurements.

Normal/Attack – Target column indicating whether the system condition is stable (“Normal”) or contains anomalous behavior (“Attack”).