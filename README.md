# SonarAnalysis

This repository has the main propuse to analyse Passive Sonar Signals with Machine Learning Algorithms and other staff

#To create folder struct do:


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
$ matlab -nodesktop -r ReadRawData; quit
```

7 - In Signal Processing Lab machines, run matlab with LofarAnalysis command to generate LofarData Files
```
$ matlab -nodesktop -r LofarAnalysis; quit
```

8 - After run these two Matlab scripts, the files should be in $OUTPUTDATAPATH, to check it
```
$ ls $OUTPUTDATAPATH
```

