#!/bin/bash

# function for create a port forward connection

#usage ./create_lps_connection.sh <local port> <remote port>


if [ $# -eq 2 ] 
then
	echo "create_lps_connection.sh"
	echo Local Port Number: $1
	echo Remote Port Number: $2
	echo "Opening Connection"
	ssh -N -f -L localhost:$1:localhost:$2 natmourajr@bastion.lps.ufrj.br
else
	echo "invalid number of arguments"
	echo "USAGE: ./create_lps_connection.sh <local port> <remote port>" 
fi


