#!/bin/bash

# function for create a port forward connection

#usage ./create_machine_struct.sh

echo "Welcome young padawan..."
echo "Creating LPS Machine Struct"

# Peterson 
./create_connection.sh 8001 satigny 8001

# Pedro
./create_connection.sh 8002 satigny 8002

# Luiza
./create_connection.sh 8003 satigny 8003
