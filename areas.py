#!/usr/bin/env python
# encoding: utf-8
"""
areas.py

Created by Zigmund Kermish on 2014-01-20.
"""

import sys
import os
import numpy as np

def load_areas(config='theo', insNum = 6.0):

	if config =='theo' or config == 'theo2' or config == 'lloro' or config == 'theo_alt1':

		#----------------------------------------------------------------------
		# This is a WRONG way of calculating the area of the MT.
		# We are forgetting the area REMOVED by the insert tubes which is
		# significant.
		#----------------------------------------------------------------------

		#Area of the SFT, based on SolidWorks drawings
		SFT_Area = 7123 #[cm**2]

		#Radius and height of MT
		R_MT = 168.4/2   #[cm]
		h_MT = 112.9     #[cm]

		#Area of MT
		MT_disk = np.pi*R_MT**2       #[cm**2]
		MT_cyl = 2*np.pi*R_MT*h_MT     #[cm**2]
		MT_Area = 2*MT_disk+MT_cyl    #[cm**2]

		insApt =   729.65877 #12 in diameter aperture in cm**2
		MT_Area -= 2*insNum*insApt
		MT_Area = MT_Area*1.3

		#Radius and height of VCS1
		R_VCS1 = 176.6/2   #[cm]
		h_VCS1 = 140.9     #[cm]

		#Area of VCS1
		VCS1_disk = np.pi*R_VCS1**2      #[cm**2]
		VCS1_cyl = 2*np.pi*R_VCS1*h_VCS1    #[cm**2]
		VCS1_Area = 2*VCS1_disk+VCS1_cyl   #[cm**2]
		VCS1_Area -= 2*insNum*insApt

		#Radius and height of VCS2
		R_VCS2 = 186.1/2  #[cm]
		h_VCS2 = 148.8    #[cm]

		#Area of VCS2
		VCS2_disk = np.pi*R_VCS2**2      #[cm**2]
		VCS2_cyl = 2*np.pi*R_VCS2*h_VCS2    #[cm**2]
		VCS2_Area = 2*VCS2_disk+VCS2_cyl   #[cm**2]
		VCS2_Area -= 2*insNum*insApt

	elif config =='ULDB' or config=='ULDB2':

		#Area of the SFT, estimate based on SolidWorks drawings of testCryostat
		SFT_Area = 2*1792.88307 #[cm**2]

		#Radius and height of MT
		MT_IR = 20.7645 #[cm]
		MT_OR = 26.0350 #[cm]
		R_MT = 26.0350   #[cm]
		h_MT = 111.0     #[cm]
		#Area of MT
		MT_disktop = np.pi*(MT_OR**2 - MT_IR**2)
		MT_diskbottom = np.pi*MT_OR**2       #[cm**2]
		MT_cyl = 2*np.pi*MT_OR*h_MT     #[cm**2]

		MT_Area = MT_disktop + MT_diskbottom + MT_cyl    #[cm**2]

		MT_Area = MT_Area*1.3

		#Radius and height of VCS1
		VCS1_OR = 27.3431   #[cm]
		VCS1_IR = 20.32000 #[cm]
		h_VCS1 = 140.78712     #[cm]

		#Area of VCS1
		VCS1_diskbottom = np.pi*VCS1_OR**2      #[cm**2]
		VCS1_disktop = np.pi*(VCS1_OR**2 - VCS1_IR**2)
		VCS1_cyl = 2*np.pi*VCS1_OR*h_VCS1    #[cm**2]
		VCS1_Area = VCS1_diskbottom + VCS1_disktop + VCS1_cyl   #[cm**2]

		#Radius and height of VCS2
		R_VCS2 = 30.31  #[cm]
		VCS2_OR = 30.31 #[cm]
		VCS2_IR = 22.225 #[cm]
		h_VCS2 = 147.066    #[cm]

		#Area of VCS2
		VCS2_diskbottom = np.pi*VCS2_OR**2      #[cm**2]
		VCS2_disktop = np.pi*(VCS2_OR**2 - VCS2_IR**2)
		VCS2_cyl = 2*np.pi*VCS2_OR*h_VCS2    #[cm**2]
		VCS2_Area = VCS2_diskbottom + VCS2_disktop + VCS2_cyl   #[cm**2]

	elif config =='TNG':

		#Area of the SFT, estimate based on SolidWorks drawings of testCryostat
		SFT_Area = 314 #[cm**2] for a 5cm ball with Volume of ~500mL

		#Area of MT from TNG Excel
		MT_can = 2827.84   #[in**2]
		MT_lid = 4013.67   #[in**2]
		MT_top = 723.58     #[in**2]
		MT_hat = 73.19     #[in**2]
		MT_Area = (MT_can + MT_lid + MT_top + MT_hat)*2.54**2   #[cm**2]

		# MT_Area = MT_Area*1.3

		#Area of VCS1 from TNG Excel
		VCS1_can = 2964.09    #[in**2]
		VCS1_lid = 4460.58    #[in**2]
		VCS1_top = 882.08     #[in**2]
		VCS1_hat = 256.91     #[in**2]
		VCS1_Area = (VCS1_can + VCS1_lid + VCS1_top + VCS1_hat)*2.54**2   #[cm**2]


		#Area of VCS2 from TNG Excel
		VCS2_can = 3361.11    #[in**2]
		VCS2_lid = 4922.28   #[in**2]
		VCS2_top = 1375.5     #[in**2]
		VCS2_hat = 497.      #[in**2]
		VCS2_Area = (VCS2_can + VCS2_lid + VCS2_top + VCS2_hat)*2.54**2   #[cm**2]


	return SFT_Area, MT_Area, VCS1_Area, VCS2_Area
