Flybrains is a project to capture and analyse iamges of drosophila larvae 

## User Guide
Firstly we need to install Python:
download from this link --> [link](https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe)

### Installing 
Make sure to add Python as a PATH variable to do so click custom installation

![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/5e1f27ca-33f5-4d86-b185-eda60e93e5fc)

Ensure all boxes are checked then click "next"

![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/652585ea-65b4-4e7c-8da2-898ff2ad743c)

Make sure "Add Python to environmental variables" is checked then click install

![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/ec055923-5f8d-4cf3-afb6-8969c891583d)

Now python should be installed

### PC Setup
Firstly download the project as a .zip file and unzip wherever you wish the program to be installed
![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/98f0bede-7e48-43a6-a77e-6c9c121ea1ea)

Now double click "install_dependencies.bat"
![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/2a57528c-5dff-471e-b57f-32ab0aa37fe6)

Now if you run "flybrains_app.py" you should be able to see the user interface

![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/060f84a9-d847-47b7-b401-35dc6c7f73e0)
![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/848c89cf-6583-4cc7-8b0f-6f51e1367be5)

or alternatively run "visualiser_app.py" if you want to process the video to make it visable without applying the AI
![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/20d1ae9f-9a4e-4e8e-a6e1-651fb01acdb9)


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


![image](https://github.com/CharlieAshdown/FlyBrains/assets/146943373/31f99be8-da8e-4b9d-b003-0096a1d5bd9d)




