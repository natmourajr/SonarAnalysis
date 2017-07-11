#!/bin/bash 

# SonarAnalysis/NoveltyDetection Package Setup Script
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

export MY_OLD_PATH=$PWD

source $SONAR_WORKSPACE/setup.sh

export PACKAGE_NAME=$OUTPUTDATAPATH/NoveltyDetection

# Folder Configuration
if [ -d "$OUTPUTDATAPATH/NoveltyDetection" ]; then
    read -e -p "Folder $OUTPUTDATAPATH/NoveltyDetection exist, Do you want to erase it? [Y,n] " yn_erase
    if [ "$yn_erase" = "Y" ]; then
    	echo
        echo "creating OUTPUTDATAPATH struct"
        echo
        rm -rf $PACKAGE_NAME
        mkdir $PACKAGE_NAME
        cd $SONAR_WORKSPACE/Packages/NoveltyDetection
        for i in $(ls -d */); do 
    		mkdir $PACKAGE_NAME/${i%%/};
    		mkdir $PACKAGE_NAME/${i%%/}/pictures_files;
    		mkdir $PACKAGE_NAME/${i%%/}/output_files;
    	done
        cd $MY_OLD_PATH
    else
    	echo
    	echo "Only Export Env Variables"
    	echo 
    fi

else
	echo
    echo "OUTPUTDATAPATH: $OUTPUTDATAPATH doesnt exists"
    echo "creating OUTPUTDATAPATH/NoveltyDetection struct"
    echo
    rm -rf $PACKAGE_NAME
    mkdir $PACKAGE_NAME
    cd $SONAR_WORKSPACE/Packages/NoveltyDetection
    for i in $(ls -d */); do 
    	mkdir $PACKAGE_NAME/${i%%/};
    	mkdir $PACKAGE_NAME/${i%%/}/pictures_files;
    	mkdir $PACKAGE_NAME/${i%%/}/output_files;
    done
    cd $MY_OLD_PATH
fi

