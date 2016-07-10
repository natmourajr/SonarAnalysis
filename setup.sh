#!/bin/bash 

# SonarAnalysis Setup Script
#
# Author: natmourajr@gmail.com
#

# Env Variables

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Ubuntu
    export SONAR_WORKSPACE=/home/natmourajr/Workspace/Doutorado/SonarAnalysis
    export INPUTDATAPATH=/home/natmourajr/Public/Marinha/Data
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    export SONAR_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/SonarAnalysis
    export INPUTDATAPATH=/Users/natmourajr/Public/Marinha/Data
fi

export OUTPUTDATAPATH=$SONAR_WORKSPACE/Results
export PYTHONPATH=$SONAR_WORKSPACE:$PYTHONPATH

export MY_PATH=$PWD

# Folder Configuration
if [ -d "$OUTPUTDATAPATH" ]; then
    read -e -p "Folder $OUTPUTDATAPATH exist, Do you want to erase it? [Y,n] " yn_erase
    if [ "$yn_erase" = "Y" ]; then
        echo "creating OUTPUTDATAPATH struct"
        rm -rf $OUTPUTDATAPATH
        mkdir $OUTPUTDATAPATH
        cd $SONAR_WORKSPACE/Packages
        for i in $(ls -d */); do mkdir $OUTPUTDATAPATH/${i%%/}; done
    fi
else
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exists"
    echo "creating OUTPUTDATAPATH struct"
    rm -rf $OUTPUTDATAPATH
    mkdir $OUTPUTDATAPATH
    cd $SONAR_WORKSPACE/Packages
    for i in $(ls -d */); do mkdir $OUTPUTDATAPATH/${i%%/}; done
fi

cd $MY_PATH

if [ -d "$OUTPUTDATAPATH/DataHandler" ]; then
	rm -rf $OUTPUTDATAPATH/DataHandler
fi
