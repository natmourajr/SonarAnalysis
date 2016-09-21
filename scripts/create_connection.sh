#!/bin/bash

# function for create a port forward connection

#usage ./create_connection.sh <local port> <remote machine name> <remote port>


if [ $# -eq 3 ] 
then
	echo "create_connection.sh"
	echo Local Port Number: $1
	echo Remote Machine Name: $2
	echo Remote Port Number: $3
	echo "Opening Connection"
	ssh -N -f -L localhost:$1:localhost:$3 natmourajr@$2.lps.ufrj.br
else
	echo "invalid number of arguments"
	echo "USAGE: ./create_connection.sh <local port> <remote machine name> <remote port>" 
fi


