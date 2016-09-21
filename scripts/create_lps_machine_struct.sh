#!/bin/bash

# function for create a port forward connection

#usage source create_lps_machine_struct.sh

# Preprocessing 
./create_lps_connection.sh 8777 8777

# Novelty Detection
./create_lps_connection.sh 8778 8778

# Stationarity Analysis
#./create_connection.sh 8779 8779

# Classification
./create_lps_connection.sh 8779 8779
