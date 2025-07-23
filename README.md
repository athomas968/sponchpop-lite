Hi both! Follow these instructions and you should be able to run SPONCHpop-lite on your laptop. You may have to log into github via your terminal before doing the following steps.

In your terminal, navigate to a folder/location where you want to keep the code. Then enter:
  git init repo-name (the repository name can be anything you like, e.g. sponchpop)
  
After that, enter 'git clone repo-name url.git' (url.git in this case will be https://github.com/athomas968/sponchpop-lite.git)
Check what branch you are in by using 'git branch -a', and then use 'git checkout jacob/daniel' to navigate to your respective branch

From here you can open VScode, and you should then be able to open the code as normal. Before you run anything, in the terminal in VScode, enter the following command:
Now you should be able to run the code! Go to 'run_it.ipynb' and there should be plots already in there.
You can change the input parameters for the disk (which includes the stellar parameters like stellar mass, radius and temperature), and from line 208, you can change the planet's initial parameters (birth location, birth time in the disk, birth mass etc. 
The switches here are to turn certain processes on and off: planetesimal driven migration (PDM), migration, pebble accretion, planetesimal accretion, gas accretion, and grain growth in the disk)

Knock or email if you're having trouble with anything!
Anna
