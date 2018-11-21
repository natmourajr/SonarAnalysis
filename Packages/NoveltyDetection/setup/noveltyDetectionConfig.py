import os
from mainConfig import CONFIG


CONFIG["PACKAGE_NAME"] = os.path.join(CONFIG["OUTPUTDATAPATH"], "NoveltyDetection")
CONFIG["PACKAGE_PATH"] = os.path.join(CONFIG["SONAR_WORKSPACE"], "Packages", "NoveltyDetection")


