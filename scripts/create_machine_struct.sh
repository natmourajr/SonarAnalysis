#!/bin/bash

# function for create a port forward connection

#usage source create_machine_struct.sh

# Preprocessing 
./create_connection.sh 8777 cessy 8777

# Novelty Detection
./create_connection.sh 8778 cessy 8778

# Stationarity Analysis
#./create_connection.sh 8779 8779

# Classification
./create_connection.sh 8779 cessy 8779


