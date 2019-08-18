import os
import platform

CONFIG = dict(osType=platform.system(), LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")

if CONFIG["osType"] == "Linux":
    CONFIG['currentUser'] = os.getenv("USER")
    CONFIG["homePath"] = os.getenv("HOME")
elif CONFIG["osType"] == "Windows":
    CONFIG['currentUser'] = os.getenv("USERNAME")
    CONFIG["homePath"] = os.getenv("UserProfile").replace("\\\\", '\\')
else:
    exit(0)

CONFIG["SONAR_WORKSPACE"] = os.path.join(CONFIG["homePath"], "Workspace", "SonarAnalysis")
CONFIG["INPUTDATAPATH"] = os.path.join(CONFIG["homePath"], "Public", "Marinha", "Data")
CONFIG["OUTPUTDATAPATH"] = os.path.join(CONFIG["SONAR_WORKSPACE"], "Results")

# Check if the results path exists
if not os.path.exists(CONFIG["OUTPUTDATAPATH"]):
    try:
        os.mkdir(CONFIG["OUTPUTDATAPATH"])
        for package in os.walk(os.path.join(CONFIG["SONAR_WORKSPACE"], "Packages")).next()[1]:
            os.mkdir(os.path.join(CONFIG["OUTPUTDATAPATH"], package))
    except Exception as ex:
        print("Error: {0}".format(ex.message))
