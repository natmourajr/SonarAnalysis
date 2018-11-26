#!/bin/bash 

# SonarAnalysis Setup Script
#
# Author: natmourajr@gmail.com
#

# Env Variables

if [[ "$USER" == "natmourajr" ]]; then
    # natmourajr user
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # Ubuntu
        export SONAR_WORKSPACE=/home/natmourajr/Workspace/Doutorado/SonarAnalysis
        export INPUTDATAPATH=/home/natmourajr/Public/Marinha/Data
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        export SONAR_WORKSPACE=/Users/natmourajr/Workspace/Doutorado/SonarAnalysis_2
        export INPUTDATAPATH=/Users/natmourajr/Workspace/Doutorado/Data/SONAR/Classification

        # For matplotlib
        export LC_ALL=en_US.UTF-8
        export LANG=en_US.UTF-8
    fi
elif [[ "$USER" == "vinicius.mello" ]]; then
    # vinicius.mello user
    if [ "$OSTYPE" == "linux-gnu" ] || ["$OSTYPE" == "GNU/Linux"]; then
        # Ubuntu
        export SONAR_WORKSPACE=/home/vinicius.mello/Workspace/SonarAnalysis
        export INPUTDATAPATH=/home/vinicius.mello/Public/Marinha/Data
    fi
fi    
export OUTPUTDATAPATH=$SONAR_WORKSPACE/Results
export PYTHONPATH=$SONAR_WORKSPACE:$PYTHONPATH

export MY_PATH=$SONAR_WORKSPACE

# Folder Configuration
if [ -d "$OUTPUTDATAPATH" ]; then
    cd $SONAR_WORKSPACE
    #read -e -p "Folder $OUTPUTDATAPATH exist, Do you want to erase it? [Y,n] " yn_erase
    #if [ "$yn_erase" = "Y" ]; then
    #    echo "creating OUTPUTDATAPATH struct"
    #    rm -rf $OUTPUTDATAPATH
    #    mkdir $OUTPUTDATAPATH
    #    cd $SONAR_WORKSPACE/Packages
    #    for i in $(ls -d */); do mkdir $OUTPUTDATAPATH/${i%%/}; done
    #fi
else
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exists"
    echo "creating OUTPUTDATAPATH struct"
    rm -rf $OUTPUTDATAPATH
    mkdir $OUTPUTDATAPATH
    cd $SONAR_WORKSPACE/Packages
    for i in $(ls -d */); do mkdir $OUTPUTDATAPATH/${i%%/}; done
fi

cd $MY_PATH
