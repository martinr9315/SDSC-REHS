#!/usr/bin/env python3

'''   _____________    ______________
-----/ ____/ __  / /_ / / ____/_  __/
----/ /   / / / / /|// / __/   / /
---/ /___/ /_/ / /  / / /___  / /
--/_____/____ /_/  /_/_____/ /_/

Created by interns in the REHS program under the
mentorship of Dr. Martin Kandes:
 - Nicholas Clark
 - Roxane Martin
 - Dustin Wu

 Run pip install -r requirements.txt to install all neccessary libraries.
 See bin/help.txt for more info.
'''

def init():
	global verbose
	global redo
	global make_visual
	global smod_add
	global load_bar
	 
	verbose = False
	redo = False
	make_visual = False
	smod_add = True
	load_bar = True