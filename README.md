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
$ matlab -nodesktop -r ReadRawData
```

7 - In Signal Processing Lab machines, run matlab with LofarAnalysis command to generate LofarData Files
```
$ matlab -nodesktop -r LofarAnalysis
```

8 - In Signal Processing Lab machines, run matlab with LofarAnalysis command to generate LofarData Files
```
$ python ReadLofarData.py
```

9 - After run these two Matlab scripts, the two new matlab files should be in $OUTPUTDATAPATH, to check it
```
$ ls $OUTPUTDATAPATH
```

10 - Some libraries are necessary, to install them do
```
$ cd $SONAR_WORKSPACE
```

11 - In this version, all analysis will be produce in Python. I suggest create a virtualenv and upgrade pip (to update pip)
```
$ pip install --upgrade pip
```

12 - In this version, all analysis will be produce in Python. I suggest create a virtualenv and install all libraries listed in requirements.txt
```
$ pip install -r requirements.txt
```

Obs: To update all packages
```
$ pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs pip install -U
```


13 - Now you can go to Packages folders and access all analysis
```
$ cd Packages
```

14 - To access a specific analysis, do
```
$ cd <Specific Analysis Name>
```

15 - Run a specific setup.sh
```
$ source setup.sh
```

16 - And
```
$ jupyter notebook
```


