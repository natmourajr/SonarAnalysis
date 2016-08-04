#!/bin/bash 

# SonarAnalysis/Classification Package Setup Script
#
# Author: natmourajr@gmail.com
#

# Check Env Variables



if [ -z "$SONAR_WORKSPACE" ]; then
	echo 
    echo "DO main setup.sh"
    echo
    return
fi  


source $SONAR_WORKSPACE/setup.sh

export MY_PATH=$PWD

export PACKAGE_OUTPUT=$OUTPUTDATAPATH/PreProcessing

# Folder Configuration
if [ -d "$OUTPUTDATAPATH/PreProcessing" ]; then
    read -e -p "Folder $OUTPUTDATAPATH/PreProcessing exist, Do you want to erase it? [Y,n] " yn_erase
    if [ "$yn_erase" = "Y" ]; then
    	echo
        echo "creating OUTPUTDATAPATH struct"
        echo
        rm -rf $PACKAGE_OUTPUT
        mkdir $PACKAGE_OUTPUT
        cd $SONAR_WORKSPACE/Packages/PreProcessing
        for i in $(ls -d */); do 
    		mkdir $PACKAGE_OUTPUT/${i%%/}; 
    		mkdir $PACKAGE_OUTPUT/${i%%/}/picts;
    		mkdir $PACKAGE_OUTPUT/${i%%/}/output_files; 
    		mkdir $PACKAGE_OUTPUT/${i%%/}/classifiers_files;
    		mkdir $PACKAGE_OUTPUT/${i%%/}/train_info_files; 
    	done
        cd $MY_PATH
    else
    	echo
    	echo "Only Export Env Variables"
    	echo 
    fi

else
	echo
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exists"
    echo "creating OUTPUTDATAPATH/Classification struct"
    echo
    rm -rf $PACKAGE_OUTPUT
    mkdir $PACKAGE_OUTPUT
    cd $SONAR_WORKSPACE/Packages/PreProcessing
    for i in $(ls -d */); do 
    	mkdir $PACKAGE_OUTPUT/${i%%/}; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/picts; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/output_files; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/classifiers_files; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/train_info_files;
    done
    cd $MY_PATH
fi

