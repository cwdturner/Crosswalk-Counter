import cv2 as cv

def click_event(event, x, y, flags, params):
	if event == cv.EVENT_LBUTTONDOWN:
		print('X: ', x, ' Y: ', y)



cap = cv.VideoCapture('TrafficVideo.mp4')
cv.namedWindow('Screen')

if cap.isOpened()==False:
	print('Error opening file')

ret, frame = cap.read()
#frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
cv.imshow('Screen', frame)

cv.setMouseCallback('Screen', click_event)

cv.waitKey(0)



cap.release()
cv.destroyAllWindows()

