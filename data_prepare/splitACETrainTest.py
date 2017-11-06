
import os
import sys

def splitACE(dataDir,testFilenames,trainPath,testPath):
    print("Usage: python splitTrainTest.py dataDir testFilenames")
    testFiles = open(testFilenames, "r").readlines()
    testFiles = [line.strip().strip(".sgm") for line in testFiles]
    print(testFiles)

    dataPath = dataDir
    fileList = os.listdir(dataPath)
    print(fileList)
    for fileItem in fileList:
        if os.path.isdir(dataPath+fileItem):continue
        print("## Processing ", fileItem)
        filename = fileItem.strip(".sgm").strip(".apf.xml")
        if filename in testFiles:
            os.rename(dataPath+fileItem, testPath+fileItem)
        else:
            os.rename(dataPath+fileItem, trainPath+fileItem)


if __name__ == "__main__":
    dataDir="D:/Code/pycharm/Sequence-Label-Attention/data/ace_en_source/"
    testFilenames="D:/Code/pycharm/Sequence-Label-Attention/data/split1.0/ACE_test_filelist"
    trainPath="D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/ace_en_experiment/train/"
    testPath="D:/Code/pycharm/Sequence-Label-Attention/data_prepare/data/ace_en_experiment/test/"
    splitACE(dataDir,testFilenames,trainPath,testPath)
