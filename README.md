# SonarAnalysis

This repository has the main propuse to analyse Passive Sonar Signals with Machine Learning Algorithms and other staff

## To create folder struct do:


1 - First clone the package lastest version
```
$ git clone https://github.com/natmourajr/SonarAnalysis.git
```

2 - Go to package folder
```
$ cd SonarAnalysis
```
3 - This package works with a couple of Environment Variables, to change their values, do
```
$ vim setup.sh
```

Edit $SONAR_WORKSPACE = analysis path, where the magic happens

4 - After change Environment variables, do
```
$ source setup.sh
```

5 - After change Environment variables, do
```
$ cd Packages
```

6 - In Signal Processing Lab machines, run matlab with ReadRawData command to generate RawData Files
```
$ matlab -nodesktop -r ReadRawData; exit
```

7 - In Signal Processing Lab machines, run matlab with LofarAnalysis command to generate LofarData Files
```
$ matlab -nodesktop -r LofarAnalysis; exit
```

8 - After run these two Matlab scripts, the two new matlab files should be in $OUTPUTDATAPATH, to check it
```
$ ls $OUTPUTDATAPATH
```

9 - Some libraries are necessary, to install them do
```
$ cd $SONAR_WORKSPACE
```

10 - In this version, all analysis will be produce in Python. I suggest create a virtualenv and install all libraries listed in requirements.txt
```
$ pip install numpy scipy matplotlib sklearn jupyter pandas
```

11 - Now you can go to Packages folders and access all analysis
```
$ cd Packages
```

12 - To access a specific analysis, do
```
$ cd Packages/<Specific Analysis Name>
```

13 - And
```
$ jupyter notebook
```


