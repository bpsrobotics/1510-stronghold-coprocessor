# the-deal
it'll knock ur socks off


##Dependancies:
Python3   
Python2.7   
NumPy    
OpenCV (cv2)  
PyNetworkTables    


##First, you gotta do some stuff. Here's what you gotta do:
Open pythonCV/mjpgToConvex.py   
Change line 17 to be whatever file path you want to use for the serialized data   
Change line 35 to be whatever video stream you're using   

## Variables at begenning of file:
debug: Just reports how long each step takes   
fileWrite: If true, will write output to fWPath   
displayProcessed: If true, will make a window with the output, can be closed with q   


## Stuff to do if you want to make it game-ready:
Modify srcImg on line 35 to be the mjpg stream   
Make it always output to one file (overwriting) so that you can view it with the dashboard (and some code)   
Make sure NetworkTables code works. I don't know if it actually works so you might have to rewrite pythonNT/NTSendArray.py in java or something (and change pythonCV/mjpgToConvex.py's serialization so java can read it)   
Change HSL and RGB constants (line 22) to match what it should be with our cameras.   
    I use HSL for the blue/green color and RGB for the white inner color, although with a threshold boundary box size comparitor (and checking for roughly 90 degree angles), you should only need HSL   



Pretty much, just run mjpgToConvex.py. If you have the RealFullField folder thing and you pass mjpgToConvex.py a value (./mjpgToConvex.py 19, for example), it'll read that number image from ../RealFullField, which is an absolute path cause i'm bad at code which you should change at line 56 for testing purposes   

Anyways, it'll then just output a serialized (python pickle) file to wherever you set the path to (my home folder by default) and then you can read that with pythonNT, which is self explanitory I think. Just have it load the serialized file (variable in it with the path), and change the IP of the NetworkTables and the name of it ("AutoAim" by default), and you should be good to go.   
