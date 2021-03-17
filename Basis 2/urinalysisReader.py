# Created by: Tom Scherr (PragmaDx, Inc.)
# Created on: January 11, 2021

# Program overview
# This software accepts an image of a urinalysis strip in a testing housing.
# (Specifically: https://www.amazon.com/gp/product/B071XTRPPT/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
# The software identifies fiducial markers for image registration that are on the paper cover for the test. It performs
# perspective transformation to project the image onto a flat surface. Next, the software identifies test pads for each
# of the 11 biochemical assays on the urinalysis strip - these are based on relative location. It performs colorimetric
# analysis based on the individual reaction. The program returns a JSON object with the resulting concentration of each
# assay, along with any statistical analysis.

# + Imports
import json
import sys
from analyzeUrinalysisImage import analyzeUrinalysis, PrintException


if __name__ == '__main__':

    # + Parameters
    # - Operations
    webServer = False
    silencePlots = True
    silencePrints = True

    # + File prefixes
    dataPathPrefix = 'data_'

    # + Load file, depending on if running locally or on server
    if webServer:

        # + Sample Node code to call Python from server
        # var outputData = [];
        # const pyProg = spawn('python', ['functionalizedProjections.py', path/To/ImageOfUrinalysisTest.jpeg, uniqueFileName]);
        #
        # pyProg.stdout.on('data', function(data)
        # {
        # console.log('Pipe data from python script ...');
        # outputData.push(data);
        # });
        #
        # pyProg.on('close', (code) = > {
        #     console.log('child process close all stdio with code ${code}');
        # // send data to browser
        # console.log(outputData.join(""));
        # res.send(outputData.join(""));
        #
        # + Example call from PHP server:
        # // + Run Python Script
        # $commandToRunPyProg = '/usr/bin/python3'. ' ' . '/path/to/urinalysisReader.py' . ' ' . 'path/To/ImageOfUrinalysisTest.jpeg' . ' ' . uniqueFileName;
        # $escapedCommandToRunPyProg = escapeshellcmd($commandToRunPyProg);
        # $PyProgOutput = shell_exec($escapedCommandToRunPyProg);


        # + Web server file storage location:
        # TODO This will need to be modified according to server file path
        resultStoragePath = '/var/www/html/Basis/App/Temp/'
        # - Retrieve filename from Python argument on server
        queryImageFileName = sys.argv[1]
        tempFileName = sys.argv[2]




    else:
        # + Local/debug datafilename:
        resultStoragePath = './TempResults/'
        tempFileName = 'temp'

        # + Test images
        queryImageFileName = './Test Images/IMG_4556.jpg'
        queryImageFileName = './Test Images/IMG_4553.jpg'
        queryImageFileName = './Test Images/IMG_4563.jpg'
        queryImageFileName = './Test Images/IMG_4558.jpg'

    # + Initialize return dictionary for results
    returnResults = {}

    # + Using try loop to catch exceptions
    try:
        # + Run test analysis algorithm
        returnResults = analyzeUrinalysis(queryImageFileName, tempFileName, resultStoragePath, silencePrints, silencePlots)
        # + Set error flag as flag
        returnResults['imgProcErrorCode'] = 0

    except Exception as imgProcErr:
        # + Handling error found during image processing
        imgProcExceptionString = PrintException()

        # + Set error flag, exception thrown, and recommendation to retake photo in dictionary
        returnResults['imgProcErrorCode'] = -1
        returnResults['exceptionString'] = imgProcExceptionString
        returnResults['recommendation'] = 'There was an error processing your photo. Please try taking another photo.'



    dataFilename = resultStoragePath + dataPathPrefix + tempFileName + '.json'
    with open(dataFilename, 'w') as outfile:
        json.dump(returnResults, outfile)


