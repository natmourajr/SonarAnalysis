#!/bin/bash 

# SonarAnalysis/NoveltyDetection Package Setup Script
#
# Author: natmourajr@gmail.com
#

# Check Env Variables

if [ -z "$OUTPUTDATAPATH" ]; then
	echo 
    echo "DO main setup.sh"
    echo
    return
fi  

export MY_PATH=$PWD

export PACKAGE_OUTPUT=$OUTPUTDATAPATH/NoveltyDetection

# Folder Configuration
if [ -d "$OUTPUTDATAPATH/NoveltyDetection" ]; then
    read -e -p "Folder $OUTPUTDATAPATH/NoveltyDetection exist, Do you want to erase it? [Y,n] " yn_erase
    if [ "$yn_erase" = "Y" ]; then
    	echo
        echo "creating OUTPUTDATAPATH struct"
        echo
        rm -rf $PACKAGE_OUTPUT
        mkdir $PACKAGE_OUTPUT
        cd $SONAR_WORKSPACE/Packages/NoveltyDetection
        for i in $(ls -d */); do 
    		mkdir $PACKAGE_OUTPUT/${i%%/}; 
    		mkdir $PACKAGE_OUTPUT/${i%%/}/picts;
    		mkdir $PACKAGE_OUTPUT/${i%%/}/output_files; 
    	done
        cd $MY_PATH
    else
    	echo
    	echo "I did nothing"
    	echo 
    fi

else
	echo
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exists"
    echo "creating OUTPUTDATAPATH/NoveltyDetection struct"
    echo
    rm -rf $PACKAGE_OUTPUT
    mkdir $PACKAGE_OUTPUT
    cd $SONAR_WORKSPACE/Packages/NoveltyDetection
    for i in $(ls -d */); do 
    	mkdir $PACKAGE_OUTPUT/${i%%/}; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/picts; 
    	mkdir $PACKAGE_OUTPUT/${i%%/}/output_files; 
    done
    cd $MY_PATH
fi

