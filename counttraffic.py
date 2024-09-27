import cv2 as cv
import numpy as np
import tensorflow as tf
import os, glob
import tensorflow_hub as hub
import math

codec = cv.VideoWriter_fourcc(*'XVID')

cars_count = []
bikes_count = []
people_count = []

'''
class_names = ['person', 'bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
	'traffic light','fire hydrant','stop sign','parking meter','bench',
	'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
	'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
	'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
	'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
	'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
	'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
	'pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse',
	'remote','keyboard','cell phone','microwave','oven','toaster','sink',
	'refrigerator','book','clock','vase','scissors','teddy bear',
	'hair drier','toothbrush'
]
'''
#Euclidean Distance Helper Function
def euclideanDistance(x1, y1, x2, y2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2)

#Area of a triangle Helper Function
def tri_area(x1, y1, x2, y2, x3, y3):
	return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1-y2)) / 2.0)

#Point inside a triangle Helper Function
def inside_tri(x1, y1, x2, y2, x3, y3, px, py):
	A = tri_area(x1, y1, x2, y2, x3, y3)
	A1 = tri_area(px, py, x2, y2, x3, y3)
	A2 = tri_area(x1, y1, px, py, x3, y3)
	A3 = tri_area(x1, y1, x2, y2, px, py)
	
	if A == A1 + A2 + A3:
		return True
	else:
		return False

#Object Class for tracking detections
class Obj:
	def __init__(self, label, x, y, ID, top, bot):
		self.label = label
		self.x = x
		self.y = y
		self.age = 0
		self.ID = ID
		self.updated = True
		self.yTop = top
		self.yBot = bot
		
	#Update the position of the object and flag that it has been updated this frame
	def update(self, x, y, top, bot):
		self.age = 0
		self.x = x
		self.y = y
		self.updated = True
		self.yTop = top
		self.yBot = bot
		
	#Increase the age of the object
	def ageUp(self):
		self.age += 1
		self.updated = False

#Tracker Class that stores objects currently being tracked and updates them using detections from the model
class Tracker:
	def __init__(self):
		self.Cars = []
		self.People = []
		self.Bikes = []
		self.nextID = 0
		
	#Add an object to the list matching its type
	def addObj(self, detect):
		tempObj = Obj(detect[0], detect[1], detect[2], self.nextID, detect[5], detect[6])
		self.nextID += 1
		if detect[0] == 'Car':
			self.Cars.append(tempObj)
		elif detect[0] == 'Person':
			self.People.append(tempObj)
		elif detect[0] == 'Bicycle':
			self.Bikes.append(tempObj)
	
	#Returns how many objects of a type haven't been updated yet
	def availObjs(self):
		carCount = 0
		personCount = 0
		bikeCount = 0
		for car in self.Cars:
			if car.updated == False:
				carCount += 1
		for person in self.People:
			if person.updated == False:
				personCount += 1
		
		for bike in self.Bikes:
			if bike.updated == False:
				bikeCount += 1
		
		return carCount, personCount, bikeCount

	def suitable_target(self, tag, x, y):
		if tag == 'Car':
			for car in self.Cars:
				if euclideanDistance(x, y, car.x, car.y) < 750 and car.updated == False:
					return True
		
		elif tag == 'Person':
			for person in self.People:
				if euclideanDistance(x, y, person.x, person.y) < 750 and person.updated == False:
					return True
		elif tag == 'Bicycle':
			for bike in self.Bikes:
				if euclideanDistance(x, y, bike.x, bike.y) < 500 and bike.updated == False:
					return True
		return False
		
	#Track the detections by either creating new objects for them or matching them to existing objects and updating those objects	
	def track(self, in_detections):
		detections = in_detections
		cCount, pCount, bCount = self.availObjs()
		removeList = []
		#Add obj if there are no available objs of that type
		#Determine the closest matching obj to the detection and update the detection
		for detect in detections:
			if detect[0] == 'Car':
				if cCount == 0:
					self.addObj(detect)
					removeList.append(detect)
				else:
					i = 0
					for car in self.Cars:
						#If closer, update detect to the index and distance of the obj
						if euclideanDistance(detect[1],detect[2], car.x, car.y) < detect[4] and car.updated == False:
							detect[4] = euclideanDistance(detect[1],detect[2], car.x, car.y)
							detect[3] = i
						i += 1
					
			elif detect[0] == 'Person':
				if pCount == 0:
					self.addObj(detect)
					removeList.append(detect)
				else:
					i = 0
					for person in self.People:
						#If closer, update detect to the index and distance of the obj
						if euclideanDistance(detect[1],detect[2], person.x, person.y) < detect[4] and person.updated == False:
							detect[4] = euclideanDistance(detect[1],detect[2], person.x, person.y)
							detect[3] = i
						i += 1
			
			elif detect[0] == 'Bicycle':
				if bCount == 0:
					self.addObj(detect)
					removeList.append(detect)
				else:
					i = 0
					for bike in self.Bikes:
						#If closer, update detect to the index and distance of the obj
						if euclideanDistance(detect[1],detect[2], bike.x, bike.y) < detect[4] and bike.updated == False:
							detect[4] = euclideanDistance(detect[1],detect[2], bike.x, bike.y)
							detect[3] = i
						i += 1
						
		#detections now has updated indexes and distances for the closest objects
		#Remove any items from detections that were added as objects
		for item in removeList:
			detections.remove(item)
		removeList.clear()
		#If there are no detections we exit
		if len(detections) == 0:
			#print('Exit at first spot')
			return
		
		#Sort detections by euclidean distance to closest obj.
		#This allows the best fitting detections to update their objects before a wrong fitting one.
		detections.sort(key=lambda x: x[4])
		
		#Go through detections, updating the corresponding object.
		for detect in detections:
			if detect[0] == 'Car':
				if self.Cars[detect[3]].updated == False:
					#Make sure no big move between cars happens
					if detect[4] < 750:
						self.Cars[detect[3]].update(detect[1],detect[2], detect[5], detect[6])
						removeList.append(detect)
					
			elif detect[0] == 'Person':
				if self.People[detect[3]].updated == False:
					self.People[detect[3]].update(detect[1],detect[2], detect[5], detect[6])
					removeList.append(detect)
			
			elif detect[0] == 'Bicycle':
				if self.Bikes[detect[3]].updated == False:
					self.Bikes[detect[3]].update(detect[1],detect[2], detect[5], detect[6])
					removeList.append(detect)
		
		for item in removeList:
			detections.remove(item)
		removeList.clear()
		#If there are no detections left, we exit
		if len(detections) == 0:
			#print('Exit at second spot')
			return
		
		#Check that of the available objects for each detection, that atleast 1 is in range of being a suitable match
		for detect in detections:
			if self.suitable_target(detect[0], detect[1], detect[2]) == False:
				self.addObj(detect)
				removeList.append(detect)
		
		#Remove any detections that did not have a suitable match
		for item in removeList:
			detections.remove(item)
		
		#Reset detections Euclidean Distances:
		for detect in detections:
			detect[4] = 999999
		#If we still have detections that did not match with an existing object, we recursively call track again on our
		#smaller detection list.
		self.track(detections)
	
	
	
	
	#Update the tracker's objects. Increasing their age and removing any object past its prime					
	def update(self):
		for car in self.Cars:
			car.ageUp()
			if car.age >= 10:
				self.Cars.remove(car)
		
		for person in self.People:
			person.ageUp()
			if person.age >= 60:
				self.People.remove(person)
			
				
		for bicycle in self.Bikes:
			bicycle.ageUp()
			if bicycle.age >= 60:
				self.Bikes.remove(bicycle)
		
	#Given the coordinates of a region update the counts if any object has entered the region		
	def count_region(self, x1, y1, x2, y2, x3, y3, x4, y4):
		global cars_count
		global bikes_count
		global people_count
		
		for car in self.Cars:
			if inside_tri(x1, y1, x2, y2, x3, y3, car.x, car.y) or inside_tri(x4, y4, x2, y2, x3, y3, car.x, car.y) or inside_tri(x1, y1, x2, y2, x3, y3, car.x, car.yTop) or inside_tri(x4, y4, x2, y2, x3, y3, car.x, car.yTop) or inside_tri(x1, y1, x2, y2, x3, y3, car.x, car.yBot) or inside_tri(x4, y4, x2, y2, x3, y3, car.x, car.yBot):
				if cars_count.count(car.ID) == 0:
					cars_count.append(car.ID)
					
		for person in self.People:
			if inside_tri(x1, y1, x2, y2, x3, y3, person.x, person.y) or inside_tri(x4, y4, x2, y2, x3, y3, person.x, person.y) or inside_tri(x1, y1, x2, y2, x3, y3, person.x, person.yTop) or inside_tri(x4, y4, x2, y2, x3, y3, person.x, person.yTop) or inside_tri(x1, y1, x2, y2, x3, y3, person.x, person.yBot) or inside_tri(x4, y4, x2, y2, x3, y3, person.x, person.yBot):
				if people_count.count(person.ID) == 0:
					people_count.append(person.ID)
					
		for bicycle in self.Bikes:
			if inside_tri(x1, y1, x2, y2, x3, y3, bicycle.x, bicycle.y) or inside_tri(x4, y4, x2, y2, x3, y3, bicycle.x, bicycle.y) or inside_tri(x1, y1, x2, y2, x3, y3, bicycle.x, bicycle.yTop) or inside_tri(x4, y4, x2, y2, x3, y3, bicycle.x, bicycle.yTop) or inside_tri(x1, y1, x2, y2, x3, y3, bicycle.x, bicycle.yBot) or inside_tri(x4, y4, x2, y2, x3, y3, bicycle.x, bicycle.yBot):
				if bikes_count.count(bicycle.ID) == 0:
					bikes_count.append(bicycle.ID)




#Draw boxes around the detected objects, and create a list of detections to be passed to the tracker
def drawBoxes(image, boxes, class_names, scores):

	detects = []
	# ['Class_Name', center point (x,y)]

	for i in range(len(boxes)):
		if class_names[i] == 3:
			boxColor = (255,0,0)
			label = 'Car'
			if scores[i] >= 0.47:
				height, width, _ = image.shape
				ymin, xmin, ymax, xmax = tuple(boxes[i])
				(left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin*height), int(ymax*height))
				image = cv.rectangle(image, (left,top), (right,bottom), boxColor, 4)
				centerX = (right - left) // 2 + left
				centerY = (bottom - top) // 2 + top
				detects.append([label, centerX, centerY, None, 999999, top, bottom])
				image = cv.circle(image, (centerX, centerY), 2, (255,255,255), 2)
			else:
				continue
			
		elif class_names[i] == 1:
			boxColor = (0,255,0)
			label = 'Person'
			if scores[i] >= 0.5:
				height, width, _ = image.shape
				ymin, xmin, ymax, xmax = tuple(boxes[i])
				(left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin*height), int(ymax*height))
				image = cv.rectangle(image, (left,top), (right,bottom), boxColor, 4)
				centerX = (right - left) // 2 + left
				centerY = (bottom - top) // 2 + top
				detects.append([label, centerX, centerY, None, 999999, top, bottom])
				image = cv.circle(image, (centerX, centerY), 2, (255,255,255), 2)
			else:
				continue
			
		elif class_names[i] == 2:
			boxColor = (0,0,255)
			label = 'Bicycle'
			if scores[i] >= 0.5:
				height, width, _ = image.shape
				ymin, xmin, ymax, xmax = tuple(boxes[i])
				(left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin*height), int(ymax*height))
				image = cv.rectangle(image, (left,top), (right,bottom), boxColor, 4)
				centerX = (right - left) // 2 + left
				centerY = (bottom - top) // 2 + top
				detects.append([label, centerX, centerY, None, 999999, top, bottom])
				image = cv.circle(image, (centerX, centerY), 2, (255,255,255), 2)
			else:
				continue
			
		#else:
		#	continue
			
		#if scores[i] < 0.5:
		#	continue
		'''
		height, width, _ = image.shape
		ymin, xmin, ymax, xmax = tuple(boxes[i])
		(left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin*height), int(ymax*height))
		image = cv.rectangle(image, (left,top), (right,bottom), boxColor, 4)
		centerX = (right - left) // 2 + left
		centerY = (bottom - top) // 2 + top
		detections.append([label, centerX, centerY, None, 999999, top, bottom])
		image = cv.circle(image, (centerX, centerY), 2, (255,255,255), 2)
		'''

	return image, detects


#Load the model from TFHub

#module_path = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'
#module_path = 'https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1'
module_path = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'
#detector = hub.load(module_path).signatures['default']
detector = hub.load(module_path)

#Read Video
cap = cv.VideoCapture('TrafficVideo.mp4')
cv.namedWindow('Screen')
out = cv.VideoWriter('output.avi', codec, 20.0, (1920,1080))

#crosswalk = [[1299, 610], [1683, 634], [615, 853], [1525, 837]]
crosswalk = [[1299, 610], [1683, 634], [615, 853], [1436, 950]]
#crosswalk2 = np.array([[1299, 610],[1683, 634],[1525, 837], [615, 853]], np.int32)
crosswalk2 = np.array([[1299, 610],[1683, 634],[1436, 950], [615, 853]], np.int32)
crosswalk2 = crosswalk2.reshape((-1, 1, 2))

bigTracker = Tracker()

if cap.isOpened()==False:
	print('Error opening file')

while cap.isOpened():
	ret, frame = cap.read()

	if not ret:
		print('Video has finished')
		break
	height, width, channels = frame.shape
	
	#Convert Frame to Input Type
	passFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	passFrame = passFrame[tf.newaxis,:]
	#Pass Frame to Model & Get Detected Objects from model
	result = detector(passFrame)
	result = {key:value.numpy() for key,value in result.items()}
	#Draw Crosswalk Region
	frame = cv.polylines(frame, [crosswalk2], True, (200, 45, 230), 2)
	#Draw squares on frame for each object and get detections
	frame, all_detections = drawBoxes(frame, result['detection_boxes'][0], result['detection_classes'][0], result['detection_scores'][0])
	bigTracker.track(all_detections)
	#Run Counter
	bigTracker.count_region(crosswalk[0][0], crosswalk[0][1], crosswalk[1][0], crosswalk[1][1], crosswalk[2][0], crosswalk[2][0], crosswalk[3][0], crosswalk[3][1])
	
	#Draw the Object Counts on screen
	frame = cv.rectangle(frame, (100,100), (300, 230), (160, 160, 160), -1)
	cv.putText(frame, f'Cars: {len(cars_count)}', (120, 130), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
	cv.putText(frame, f'People: {len(people_count)}', (120, 170), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
	cv.putText(frame, f'Bikes: {len(bikes_count)}', (120, 210), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
	
	#Write frame to screen and to file
	cv.imshow('Screen', frame)
	out.write(frame)
	
	#Update the tracker' objects
	bigTracker.update()
	
	#
	key = cv.waitKey(1)
	if key == ord('q'):
		break
	if key == ord('d'):
		cv.imwrite('test_im1.jpg', frame)
	#Check Existing Cars
	if key == ord('r'):
		for car in bigTracker.Cars:
			print('ID ', car.ID, ' X ', car.x, ' Y ', car.y)
		print(len(all_detections))
		print()
	
#Close out the program	
out.release()		
cap.release()
cv.destroyAllWindows()

