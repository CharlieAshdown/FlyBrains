Flybrains is a project to capture and analyse iamges of drosophila larvae 

## Physical models
This system is made up of a physcial enclosure to collect larvae data. These models are available for download here: [models](https://a360.co/4b0MJiT)

## User Guide
Firstly we need to install Python:
download from this link --> [link](https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe)

### Installing 
Make sure to add Python as a PATH variable to do so click custom installation
![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/dd678b18-f63f-4cc1-b53e-76b49137931e)


Ensure all boxes are checked then click "next"

![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/a594bbca-79c5-4d95-88e6-b7dbfef386fb)


Make sure "Add Python to environmental variables" is checked then click install

![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/8a38609e-f0c6-4ed2-8d89-5c1e581824b1)


Now python should be installed

### PC Setup
Firstly download the project as a .zip file and unzip wherever you wish the program to be installed
![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/12d7147c-3ba8-422b-834e-9068f812cf39)


Now double click "install_dependencies.bat"

Now if you run "flybrains_app.py" you should be able to see the user interface

![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/dae27dd7-03a8-451f-9c3e-067516c611e1)


or alternatively run "visualiser_app.py" if you want to process the video to make it visable without applying the AI
![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/622b2d94-b9a9-41c1-986b-938115ad9670)



### Using flybrains_app.py
This app is designed to put a nice, easy to control interface between the user and the incredibly ugly code.

The speed and rotational speed (speed at which the larvae rotates) are calculated by averageing across a number of frames. 
The array length slider simply changes the number of frames to which the averages are taken. The default value to which is 10.

The larvae acceptance threshold determines to which certainty a detected object is considered a larvae and not just noise. 
The default for this is 0.9 this value has been tested to be proven the optimal value.

The find file button when clicked will allow the user to the folder in which contains the video they want outputted. 
This will be the format the video is outputted from the physical system software.
The user can either click the button to search up the folder(s) they want to process or simply drag the folders they want to process into the find file button which will have the same effect.

The save video button when checked will save the processed video.
The play video button when checked will play the video after it has been processed.
The create csv file button when checked will output all recorded data to a csv file.

All outputted content is saved in the same folder as the video.


![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/dae27dd7-03a8-451f-9c3e-067516c611e1)

## Using visualiser_app.py
Visualiser app allows you to convert the outputted .h264 file from the system into different viewable formats

You can drag and drop as many files containing videos as you want into the program and by selecting different parts of the system it will process them differently 

The single channel button if ticked reduces the video to a single colour channel of the user's choice

The save images button allows the user to save the individual frames of the image as well as processing the video. This makes frame by frame analysis easier.

The brighten button when ticked utilised otsu's binarisation to brighten the video to be easier to view by users. 

![image](https://github.com/CharlieAshdown/Flybrains/assets/146943373/622b2d94-b9a9-41c1-986b-938115ad9670)
